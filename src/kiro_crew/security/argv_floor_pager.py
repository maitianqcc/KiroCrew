"""The ``git grep`` pager read of the self-kill floor.

``git grep -O<cmd>`` (and the other places git takes its pager from) runs ``<cmd>``
on the matching files, so a pager that kills the product is a self-kill that no argv
position shows. :func:`_git_grep_pager_commands` returns every pager command a
``git grep`` in a text can run; ``argv_floor_self._is_self_kill`` asks the same kill
check about each one. This module reads the shell reader and the argv rules below it
and imports no other part of the argv floor.
"""

from __future__ import annotations

import shlex
from typing import NamedTuple

from . import argv_rules as _argv_rules
from . import shell_commands as _shell_commands
from . import shell_normalizer as _shell_normalizer

#: git's global options (before the subcommand) whose value is the NEXT word, so that
#: word is skipped rather than read as the subcommand.
_GIT_GLOBAL_VALUE_OPTIONS = frozenset(
    {
        *("-C", "-c", "--git-dir", "--work-tree", "--namespace"),
        *("--super-prefix", "--config-env", "--attr-source"),
    }
)
#: git's global options that, written bare, print or show something and exit
#: without running the subcommand after them.
_GIT_GLOBAL_EXIT_OPTIONS = frozenset({"--exec-path", "--version", "--help", "-v", "-h"})
#: The values git reads as boolean false (case-insensitive; an empty value too).
_GIT_FALSE = frozenset({"false", "no", "off", "0", ""})
#: The values git reads as boolean true (case-insensitive).
_GIT_TRUE = frozenset({"true", "yes", "on", "1"})
#: ``git grep`` short options that take a VALUE (``builtin/grep.c``): a pattern
#: (``-e``), a file (``-f``), a count (``-m``) or a context size (``-A``/``-B``/
#: ``-C``). In a cluster the rest of the word is that value, else it is the next
#: word, so either way it is never a pager. Case-sensitive: ``-E`` and ``-F`` are
#: flags that take none.
_GIT_GREP_VALUE_SHORT = frozenset("efmABC")
#: The long options of the same kind. git also accepts a unique prefix of a long
#: name, so a word is matched as a prefix of one of these.
_GIT_GREP_VALUE_LONG = (
    "--regexp",
    "--file",
    "--max-count",
    "--max-depth",
    "--threads",
    "--context",
    "--after-context",
    "--before-context",
)


#: The other ``git grep`` long options that begin ``--o``. git takes a long option's
#: unique prefix only, so a prefix shared with one of these (``--o``) is ambiguous and
#: git refuses the command instead of running a pager.
_GIT_GREP_OPEN_RIVALS = ("--or", "--only-matching")


def _grep_arg_pager(word: str) -> "tuple[str, str]":
    """How ``git grep`` parses *word*, as submitted: ``(role, pager)``.

    Roles: ``pager`` with the command git would run as its pager (``""`` for a bare
    ``-O``, which asks for the default pager); ``no_pager`` for
    ``--no-open-files-in-pager`` (or a prefix git takes for it, from ``--no-op``),
    which clears an earlier ``-O``; ``skip_next`` when *word* is an option
    whose value is the FOLLOWING word (so that word is not read as a pager); ``plain``
    otherwise. Only ``-O`` / ``--open-files-in-pager`` carry a pager, and always GLUED
    to the option, so a pager is never a separate word. A short cluster is walked one
    letter at a time, the way git's parse-options walks it: ``O`` is the pager only
    before the first letter that takes a value, since everything after that letter
    is its value.
    """
    if word.startswith("--"):
        name, _sep, value = word.partition("=")
        if not _sep and len(name) >= len("--no-op") and "--no-open-files-in-pager".startswith(name):
            return ("no_pager", "")
        if (
            len(name) > 2
            and "--open-files-in-pager".startswith(name)
            and not any(rival.startswith(name) for rival in _GIT_GREP_OPEN_RIVALS)
        ):
            return ("pager", value)
        if not _sep and len(name) > 2 and any(o.startswith(name) for o in _GIT_GREP_VALUE_LONG):
            return ("skip_next", "")
        return ("plain", "")
    if word.startswith("-") and len(word) > 1:
        j = 1
        while j < len(word) and word[j].isascii() and word[j].isalnum():
            if word[j] == "O":
                return ("pager", word[j + 1 :])
            if word[j] in _GIT_GREP_VALUE_SHORT:
                return ("plain", "") if j + 1 < len(word) else ("skip_next", "")
            j += 1
    return ("plain", "")


class _Expansion(NamedTuple):
    """The words one array expansion gives a ``git grep``."""

    words: "tuple[str, ...]"


def _scan_grep_args(
    args: "list[str | _Expansion]",
    memo: "dict[tuple[int, bool], tuple[tuple[str, ...], bool]] | None" = None,
) -> "list[str]":
    """The pager commands a ``git grep`` runs from its own arguments *args*.

    Each argument is a real word (a ``str``) or an :class:`_Expansion`, the words an
    array expansion put there. A value-taking option's value is skipped, so
    ``-e '-O<cmd>'`` reads ``-O<cmd>`` as a pattern, not a pager. The LAST ``-O``
    wins, as in git, so ``-O<kill> -Ocat`` runs only ``cat``, and a real
    ``--no-open-files-in-pager`` clears it. An inserted word is read where it stands
    but trusted less, because the array the floor read may not be the one bash
    expands: its value option never swallows the real word after it, and its ``-O``
    neither replaces nor is replaced by a real one. ``--`` is not treated as ending
    option parsing here: the issue names ``-O`` placed after the pathspec, and a word
    that both begins ``-O`` and spells a product kill is that bypass, not a real
    pathspec.

    An expansion object is scanned once per entry state (in *memo*, which a caller
    shares across a text's commands), and the caller hands the same object for
    every reference with the same text, so repeating one array reference costs the
    reference's length, not its length times the array's size.
    """
    last = ""
    inserted: "dict[str, None]" = {}
    if memo is None:
        memo = {}
    # What the previous word left pending: ``any`` (a real value option swallows the
    # next word), ``inserted`` (an inserted one swallows only an inserted word), or
    # nothing.
    pending = ""
    for arg in args:
        if isinstance(arg, str):
            if pending == "any":
                pending = ""
                continue
            role, pager = _grep_arg_pager(arg)
            pending = "any" if role == "skip_next" else ""
            if role == "pager":
                last = pager
            elif role == "no_pager":
                last = ""
            continue
        # Keyed by identity: hashing the words would read the whole array again.
        key = (id(arg), bool(pending))
        if key not in memo:
            memo[key] = _scan_expansion(arg.words, bool(pending))
        pagers, leaves_pending = memo[key]
        inserted.update(dict.fromkeys(pagers))
        pending = "inserted" if leaves_pending else ""
    return [command for command in (last, *inserted) if command.strip()]


def _scan_expansion(words: "tuple[str, ...]", swallowed: bool) -> "tuple[tuple[str, ...], bool]":
    """``(pagers, leaves a value option pending)`` for one expansion's *words*.

    *swallowed*: the word before the expansion was a value option, so the first
    inserted word is its value.
    """
    pagers: "dict[str, None]" = {}
    pending = swallowed
    for word in words:
        if pending:
            pending = False
            continue
        role, pager = _grep_arg_pager(word)
        if role == "skip_next":
            pending = True
        elif role == "pager":
            pagers[pager] = None
    return tuple(pagers), pending


#: The characters a backslash escapes inside double quotes; before any other it is
#: a literal backslash.
_DOUBLE_QUOTE_ESCAPES = frozenset('$`"\\\n')


def _shell_word(word: "_shell_commands.ShellWord") -> "tuple[str, list[int]]":
    """The text the shell passes for *word*, with each character's quoting.

    Quote syntax (an opening or closing quote) is removed, an escape pair gives the
    escaped character (a backslash-newline gives nothing, and inside double quotes a
    backslash before any other character stays), all as the shared quote machine
    marked the steps. The quoting is 0 unquoted, 1 literal (single-quoted or
    escaped), 2 double-quoted.
    """
    chars: "list[str]" = []
    quoting: "list[int]" = []
    for step in word.steps:
        if len(step.text) == 2:
            if step.char == "\n":
                continue
            kept = (
                step.text
                if step.state == 2 and step.char not in _DOUBLE_QUOTE_ESCAPES
                else step.char
            )
            chars.append(kept)
            quoting.extend([1] * len(kept))
        elif step.char in "'\"" and (step.active or step.state == 0):
            continue
        else:
            chars.append(step.char)
            quoting.append(step.state)
    return "".join(chars), quoting


def _git_config_value(setting: str) -> "tuple[str, bool, str]":
    """``(key folded, has a value, value)`` of one ``key=value`` config setting."""
    key, has_value, value = setting.partition("=")
    return key.lower(), bool(has_value), value


def _env_config_settings(env: "dict[str, str]") -> "list[str]":
    """The ``key=value`` settings git reads from *env*'s config variables.

    ``GIT_CONFIG_PARAMETERS`` holds single-quoted ``'key=value'`` or ``'key'='value'``
    entries; ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_<n>`` / ``GIT_CONFIG_VALUE_<n>``
    hold numbered pairs, of which git reads the first COUNT. Only the variables
    present are read, so a huge COUNT costs nothing. A value that cannot be parsed
    gives nothing, as git refuses it too.
    """
    settings: "list[str]" = []
    parameters = env.get("GIT_CONFIG_PARAMETERS")
    if parameters:
        try:
            settings.extend(shlex.split(parameters))
        except ValueError:
            pass
    count = env.get("GIT_CONFIG_COUNT", "")
    if count.isascii() and count.isdigit():
        pairs: "list[tuple[int, str]]" = []
        for name, key in env.items():
            index = name[len("GIT_CONFIG_KEY_") :]
            if name.startswith("GIT_CONFIG_KEY_") and index.isascii() and index.isdigit():
                if int(index) < int(count) and f"GIT_CONFIG_VALUE_{index}" in env:
                    pairs.append((int(index), f"{key}={env[f'GIT_CONFIG_VALUE_{index}']}"))
        settings.extend(setting for _index, setting in sorted(pairs))
    return settings


def _is_assignment(word: str) -> bool:
    """True if *word* is a shell ``NAME=value`` assignment word."""
    name, has_value, _value = word.partition("=")
    return (
        bool(has_value)
        and bool(name)
        and name.isascii()
        and not name[0].isdigit()
        and name.replace("_", "a").isalnum()
    )


def _shell_reads_stdin(args: "list[str]") -> bool:
    """True if a shell given *args* reads its script from standard input.

    It does unless a ``-c`` (alone or in a short-option cluster) names the script or
    a word that is not an option names a script file. ``-o`` / ``-O`` / ``+o`` /
    ``+O`` and ``--rcfile`` / ``--init-file`` take the next word as their value;
    after ``-s`` or ``--`` every word is a positional argument.
    """
    k = 0
    while k < len(args):
        arg = args[k]
        if arg in ("-s", "--"):
            return True
        if arg in ("-o", "-O", "+o", "+O", "--rcfile", "--init-file"):
            k += 2
            continue
        if arg.startswith("--"):
            k += 1
            continue
        if arg[:1] in "-+" and len(arg) > 1:
            if arg[0] == "-" and "c" in arg[1:]:
                return False
            if "s" in arg[1:]:
                return True
            k += 1
            continue
        return False  # a script file: the here-document is its input, not its code
    return True


def _git_grep_pager_commands(source: str) -> "list[str]":
    """The pager commands every ``git grep`` in *source* can run, for the kill check.

    ``git grep`` runs its pager on the matching files, so a pager that kills the
    product is a self-kill no argv position shows. Parsed the way git parses it, one
    invocation at a time and only from what that simple command itself shows:

    * the last ``-O`` / ``--open-files-in-pager=`` among grep's own arguments, with
      each literal array reference there expanded from the arrays the text's earlier
      commands set (:class:`_TextState`);
    * unless paging is off (``-P`` / ``--no-pager``, or a false ``pager.grep``
      without ``-p``), the ``GIT_PAGER=`` / ``PAGER=`` assignments before ``git``
      and the last ``core.pager`` and ``pager.grep`` git reads, from ``-c``, from
      ``--config-env`` naming a variable assigned before ``git``, or from
      ``GIT_CONFIG_PARAMETERS`` / ``GIT_CONFIG_COUNT`` assignments before ``git``;
    * an ``alias.<name>`` set by ``-c`` on the same command that resolves, alias by
      alias, to ``grep`` makes ``git <name>`` a grep, with the aliases' own words
      read first.

    A ``git grep`` run by ``find -exec`` is read the same way, and so is every
    command of a here-document fed to a shell that reads its script from standard
    input (``bash <<EOF``, ``bash -s <<EOF``, ``bash -o pipefail <<EOF``), at any
    depth: each body is read once, as its own text, with arrays of its own.

    The commands and their words come from the shared reader
    (:func:`shell_commands.read_commands`, over :func:`_iter_shell_chars`) after
    ``$'...'`` is decoded as every floor reader decodes it, so a quoted ``;`` or
    ``|`` inside an argument does not cut one ``git grep`` in two. Each distinct
    command is returned once.
    """
    commands: "dict[str, None]" = {}
    texts = [source]
    while texts:
        text = _shell_normalizer._decode_shell_quoted_literals(texts.pop())
        if "git" not in text.lower():
            continue  # no ``git`` word can be read from it, at any depth
        state = _TextState()
        opener: "tuple[str, bool] | None" = None
        for command in _shell_commands.read_commands(text, heredocs=True):
            words = [_shell_word(word) for word in command.argv]
            sub = [value for value, _quoting in words]
            if opener is not None:
                # The elements of ``name=( ... )``: the reader ends the opener at
                # ``(`` and this command at ``)``.
                state.assign_list(opener[0], sub, opener[1])
                opener = None
                continue
            parts = [_assignment_parts(word) for word in sub]
            if sub and all(parts):
                for part in parts[:-1]:
                    state.assign_word(*part)  # type: ignore[misc]
                name, index, appends, value = parts[-1]  # type: ignore[misc]
                if command.separator == "(" and index is None and not value:
                    opener = (name, appends)
                else:
                    state.assign_word(name, index, appends, value)
                continue
            # The program after any wrapper (``sudo``, ``env``, ``timeout 5`` ...) and
            # ``VAR=x`` words, the same step-over the argv rules use.
            found = _argv_rules._program_args(sub)
            if found is None:
                continue
            start = len(sub) - len(found[1])
            program = found[0].lower()
            if (
                command.heredocs
                and program in _shell_normalizer._NESTED_SHELL_PROGRAMS
                and program != "busybox"
                and _shell_reads_stdin(found[1])
            ):
                texts.extend(doc.body for doc in command.heredocs)
            invocations = [words]
            if program == "find":
                invocations = [words[start + a : start + b] for a, b in _find_exec_spans(found[1])]
            for invocation in invocations:
                commands.update(dict.fromkeys(_git_grep_invocation_pagers(invocation, state)))
    return list(commands)


def _find_exec_spans(args: "list[str]") -> "list[tuple[int, int]]":
    """``(start, end)`` in *args* of each ``-exec`` / ``-execdir`` / ``-ok`` /
    ``-okdir`` command of a ``find``, as :func:`argv_rules._find_exec_argvs` reads it."""
    spans: "list[tuple[int, int]]" = []
    for index, token in enumerate(args):
        if token in _argv_rules._FIND_EXEC:
            end = next(
                (j for j in range(index + 1, len(args)) if args[j] in (";", "+", "\\;")),
                len(args),
            )
            if end > index + 1:
                spans.append((index + 1, end))
    return spans


def _git_grep_invocation_pagers(
    words: "list[tuple[str, list[int]]]", state: "_TextState"
) -> "list[str]":
    """The pager commands one simple command's ``git grep`` can run, else ``[]``."""
    sub = [value for value, _quoting in words]
    found = _argv_rules._program_args(sub)
    # The program name is folded: a case-insensitive filesystem runs ``GIT``.
    if found is None or found[0].lower() != "git":
        return []
    i = len(sub) - len(found[1])
    # Every assignment before ``git`` is in its environment, whether it leads the
    # command or follows ``env`` or ``sudo``.
    env = dict(word.partition("=")[::2] for word in sub[: i - 1] if _is_assignment(word))
    env_pagers = [env[name] for name in ("GIT_PAGER", "PAGER") if name in env]
    settings = _env_config_settings(env)
    paging = ""  # the last of -P / --no-pager ("off") and -p / --paginate ("on")
    grep_args: "list[str] | None" = None
    while i < len(sub):
        token = sub[i]
        if token in _GIT_GLOBAL_EXIT_OPTIONS:
            return []  # git runs ``version`` / ``help`` or prints a path, never grep
        if token in ("-P", "--no-pager", "-p", "--paginate"):
            # ``-P`` makes git set ``GIT_PAGER=cat``; ``-p`` pages whatever
            # ``pager.grep`` says.
            paging = "off" if token in ("-P", "--no-pager") else "on"
        elif token.startswith("--config-env="):
            settings.extend(_config_env_setting(token[len("--config-env=") :], env))
        elif token in _GIT_GLOBAL_VALUE_OPTIONS:
            if i + 1 < len(sub) and token == "-c":
                settings.append(sub[i + 1])
            elif i + 1 < len(sub) and token == "--config-env":
                settings.extend(_config_env_setting(sub[i + 1], env))
            i += 1
        elif not token.startswith("-"):
            grep_args = _grep_subcommand(token, settings)
            i += 1
            break  # the subcommand: a grep, an alias of one, or another
        i += 1
    if grep_args is None:
        return []
    pager_values: "dict[str, tuple[bool, str]]" = {}
    for setting in settings:
        key, has_value, value = _git_config_value(setting)
        if key in ("core.pager", "pager.grep"):
            pager_values[key] = (has_value, value)  # the last one wins, as in git
    default_pagers = list(env_pagers)
    if "core.pager" in pager_values:
        default_pagers.append(pager_values["core.pager"][1])
    grep_paged = paging != "off"
    # ``pager.grep`` is read only when no -p / -P said otherwise.
    if not paging and "pager.grep" in pager_values:
        has_value, value = pager_values["pager.grep"]
        folded = value.strip().lower()
        if has_value and folded in _GIT_FALSE:
            grep_paged = False  # a boolean false: grep's paging is off
        elif has_value and folded not in _GIT_TRUE:
            default_pagers.append(value)  # a command: grep's own pager
    args: "list[str | _Expansion]" = list(grep_args)
    for value, quoting in words[i:]:
        expansion = state.expansion(value, quoting) if "$" in value else None
        args.append(value if expansion is None else expansion)
    pagers = _scan_grep_args(args, state.scans)
    if grep_paged:
        pagers += default_pagers
    return [pager for pager in pagers if pager.strip()]


def _assignment_parts(word: str) -> "tuple[str, int | None, bool, str] | None":
    """``(name, index or None, appends, value)`` of a shell assignment word, else None.

    ``a=x``, ``a+=x``, ``a[3]=x`` and ``a[3]+=x``; an array opener (``a=(`` /
    ``a+=(``) arrives as ``a=`` / ``a+=`` with an empty value.
    """
    head, has_value, value = word.partition("=")
    if not has_value:
        return None
    appends = head.endswith("+")
    if appends:
        head = head[:-1]
    index: "int | None" = None
    if head.endswith("]") and "[" in head:
        head, _sep, subscript = head[:-1].partition("[")
        if not (subscript.isascii() and subscript.isdigit()):
            return None
        index = int(subscript)
    if not head or not head.isascii() or head[0].isdigit():
        return None
    if not head.replace("_", "a").isalnum():
        return None
    return head, index, appends, value


class _TextState:
    """What every ``git grep`` of one shell text shares: the literal arrays its
    commands have set so far (each a map of index to element), each array's
    expansions per version, and the scan of each expansion per entry state.

    The floor reads arrays in its own pass over the text's commands, the way bash
    runs them in order: the last assignment wins, ``a+=(...)`` appends, ``a[N]=x``
    sets one index, and ``a=x`` on an array sets element 0."""

    def __init__(self) -> None:
        self.arrays: "dict[str, dict[int, str]]" = {}
        self.versions: "dict[str, int]" = {}
        self.elements: "dict[tuple[str, int, int | str, int], list[str]]" = {}
        self.expansions: "dict[tuple[object, ...], _Expansion | None]" = {}
        self.scans: "dict[tuple[int, bool], tuple[tuple[str, ...], bool]]" = {}

    def _changed(self, name: str) -> None:
        self.versions[name] = self.versions.get(name, 0) + 1

    def assign_list(self, name: str, elements: "list[str]", appends: bool) -> None:
        """``name=(elements)``, or ``name+=(elements)`` when *appends*."""
        current = self.arrays.setdefault(name, {}) if appends else {}
        start = max(current) + 1 if current else 0
        current.update((start + k, element) for k, element in enumerate(elements))
        self.arrays[name] = current
        self._changed(name)

    def assign_word(self, name: str, index: "int | None", appends: bool, value: str) -> None:
        """``name[index]=value`` (``+=`` appends to that element). Without an index only
        an existing array changes, at element 0; a plain scalar is not tracked."""
        if index is None:
            if name not in self.arrays:
                return
            index = 0
        current = self.arrays.setdefault(name, {})
        current[index] = current.get(index, "") + value if appends else value
        self._changed(name)

    def selected(self, name: str, selector: "int | str", state: int) -> "list[str]":
        """The words reference ``${name[selector]}`` gives, quoted per *state*."""
        key = (name, self.versions.get(name, 0), selector, state)
        if key not in self.elements:
            array = self.arrays[name]
            if selector in ("@", "*"):
                values = [array[index] for index in sorted(array)]
            else:
                values = [array.get(int(selector), "")]
            if state == 0:
                words = [part for value in values for part in value.split()]
            elif selector == "*":
                words = [" ".join(values)]
            else:
                words = values
            self.elements[key] = words
        return self.elements[key]

    def expansion(self, value: str, quoting: "list[int]") -> "_Expansion | None":
        """The words bash makes of *value* by expanding its array references, else
        None when it has none the floor can read. Cached per text and per version of
        every array it names."""
        refs = [ref for ref in _array_refs(value, quoting) if ref[2] in self.arrays]
        if not refs:
            return None
        key = (value, tuple(quoting), tuple(self.versions.get(ref[2], 0) for ref in refs))
        if key not in self.expansions:
            words = [""]
            at = 0
            for start, end, name, selector in refs:
                elements = self.selected(name, selector, quoting[start])
                words[-1] += value[at:start]
                if elements:
                    words[-1] += elements[0]
                    words.extend(elements[1:])
                at = end
            words[-1] += value[at:]
            self.expansions[key] = _Expansion(tuple(words))
        return self.expansions[key]


def _name_end(value: str, at: int) -> int:
    """The end of the shell variable name that starts at *at* (``at`` when none)."""
    end = at
    while end < len(value) and (
        value[end] == "_" or (value[end].isascii() and value[end].isalnum())
    ):
        end += 1
    if end > at and value[at].isdigit():
        return at
    return end


def _array_refs(value: str, quoting: "list[int]") -> "list[tuple[int, int, str, int | str]]":
    """``(start, end, name, selector)`` of each variable reference in *value* that
    bash expands (not single-quoted or escaped): ``$a`` and ``${a}`` select element
    0, ``${a[N]}`` element N, ``${a[@]}`` / ``${a[*]}`` every element. One pass."""
    refs: "list[tuple[int, int, str, int | str]]" = []
    i = 0
    while i < len(value):
        if value[i] != "$" or quoting[i] == 1:
            i += 1
            continue
        if value[i + 1 : i + 2] != "{":
            end = _name_end(value, i + 1)
            if end > i + 1:
                refs.append((i, end, value[i + 1 : end], 0))
                i = end
            else:
                i += 1
            continue
        end = _name_end(value, i + 2)
        name = value[i + 2 : end]
        selector: "int | str" = 0
        if end < len(value) and value[end] == "[":
            close = end + 1
            while close < len(value) and value[close].isascii() and value[close].isdigit():
                close += 1
            subscript = value[end + 1 : close]
            if not subscript and value[close : close + 1] in ("@", "*"):
                subscript = value[close]
                close += 1
            if not subscript or value[close : close + 1] != "]":
                i += 1
                continue
            selector = subscript if subscript in ("@", "*") else int(subscript)
            end = close + 1
        if name and value[end : end + 1] == "}":
            refs.append((i, end + 1, name, selector))
            i = end + 1
        else:
            i += 1
    return refs


def _config_env_setting(spec: str, env: "dict[str, str]") -> "list[str]":
    """The ``key=value`` setting ``--config-env=<key>=<var>`` makes, when *env*
    (the assignments before ``git`` on the same command) holds ``<var>``."""
    key, _sep, var = spec.partition("=")
    return [f"{key}={env[var]}"] if var in env else []


def _grep_subcommand(token: str, settings: "list[str]") -> "list[str] | None":
    """The words ``git grep`` gets before its own arguments when the subcommand is
    *token*, else None: ``[]`` for ``grep``, or the words of the ``alias.<name>``
    chain set in *settings* that ends at ``grep``. Alias names are matched folded,
    as git folds config keys; an alias's words are split as git splits them. A shell
    alias (``!...``), an unparsable one, or a chain that loops is not a grep."""
    aliases: "dict[str, str]" = {}
    for setting in settings:
        key, has_value, value = _git_config_value(setting)
        if has_value and key.startswith("alias."):
            aliases[key[len("alias.") :]] = value  # the last one wins
    prefix: "list[str]" = []
    seen: "set[str]" = set()
    while token != "grep":
        name = token.lower()
        if name in seen or name not in aliases:
            return None
        seen.add(name)
        try:
            alias_words = shlex.split(aliases[name])
        except ValueError:
            return None
        if not alias_words:
            return None
        token, prefix = alias_words[0], alias_words[1:] + prefix
    return prefix
