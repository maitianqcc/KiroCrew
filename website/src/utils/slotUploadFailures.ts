/**
 * Attachment failures waiting for their chat to come on screen.
 *
 * An upload or capture that fails after the person switched chats belongs to the
 * chat it started in, not the one on screen. It is held here, outside the
 * transcript (no history rebuild can drop or reorder it), and the chat page moves
 * it into its upload banner when that chat is next shown. Module scope, so a
 * failure that lands after the page unmounted is still there when it remounts.
 */

const waiting = new Map<string, string>()

/** Hold `message` for `slot`; a later failure for the same chat replaces it. */
export function holdSlotUploadFailure(slot: string, message: string): void {
  // shortcut: one entry per chat, never evicted; a deleted chat's entry stays
  // until reload. Upgrade to a bounded table if chats are created in bulk.
  waiting.delete(slot)
  waiting.set(slot, message)
}

/** The failure held for `slot`, removed so it is shown once. */
export function takeSlotUploadFailure(slot: string | null | undefined): string {
  if (!slot) return ''
  const message = waiting.get(slot) ?? ''
  waiting.delete(slot)
  return message
}

export function clearSlotUploadFailures(): void {
  waiting.clear()
}
