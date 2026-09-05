import { useEffect, useMemo, useState } from "react";
import {
  CheckSquare,
  Copy,
  Download,
  History,
  Pencil,
  Pin,
  PinOff,
  Search,
  ShieldCheck,
  Trash2,
  Volume2,
} from "lucide-react";
import type { Settings, Transcript } from "../types";

type ExportFormat = "json" | "csv" | "txt";

type Props = {
  history: Transcript[];
  settings: Settings;
  onClear: () => void;
  onCopy: (item: Transcript) => void;
  onCopyMany: (items: Transcript[]) => Promise<boolean>;
  onPin: (item: Transcript) => void;
  onBulkPin: (ids: string[], pinned: boolean) => Promise<boolean>;
  onBulkDelete: (ids: string[]) => Promise<boolean>;
  onExport: (ids: string[], format: ExportFormat) => Promise<boolean>;
  onEdit: (id: string, text: string) => Promise<boolean>;
  historySectionRef: (node: HTMLElement | null) => void;
};

export default function HistorySection({
  history,
  settings,
  onClear,
  onCopy,
  onCopyMany,
  onPin,
  onBulkPin,
  onBulkDelete,
  onExport,
  onEdit,
  historySectionRef,
}: Props) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkAction, setBulkAction] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("txt");
  const filteredHistory = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    return history.filter(
      (item) =>
        !normalizedQuery ||
        item.text.toLocaleLowerCase().includes(normalizedQuery) ||
        item.language.toLocaleLowerCase().includes(normalizedQuery),
    );
  }, [history, query]);
  const selectedItems = useMemo(
    () => history.filter((item) => selectedIds.has(item.id)),
    [history, selectedIds],
  );
  const allVisibleSelected =
    filteredHistory.length > 0 &&
    filteredHistory.every((item) => selectedIds.has(item.id));

  useEffect(() => {
    const availableIds = new Set(history.map((item) => item.id));
    setSelectedIds((selected) => {
      const next = new Set([...selected].filter((id) => availableIds.has(id)));
      return next.size === selected.size ? selected : next;
    });
  }, [history]);

  const saveEdit = async () => {
    if (!editingId || !editingText.trim()) return;
    if (await onEdit(editingId, editingText)) {
      setEditingId(null);
      setEditingText("");
    }
  };
  const toggleItem = (id: string) => {
    setSelectedIds((selected) => {
      const next = new Set(selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const toggleVisibleItems = () => {
    setSelectedIds((selected) => {
      const next = new Set(selected);
      if (allVisibleSelected)
        filteredHistory.forEach((item) => next.delete(item.id));
      else filteredHistory.forEach((item) => next.add(item.id));
      return next;
    });
  };
  const runBulkAction = async (
    name: string,
    action: () => Promise<boolean>,
  ) => {
    if (bulkAction) return;
    setBulkAction(name);
    try {
      await action();
    } finally {
      setBulkAction(null);
    }
  };
  const clearSelection = () => setSelectedIds(new Set());

  return (
    <section ref={historySectionRef} className="history-section" id="history">
      <div className="history-heading">
        <div>
          <History size={18} />
          <h3>Recent transcripts</h3>
          <span>
            {settings.historyRetention === "never"
              ? "Not saved"
              : `Last ${settings.historyRetention}`}
          </span>
        </div>
        {history.length > 0 && (
          <button type="button" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      <label className="history-search">
        <Search size={14} />
        <span className="sr-only">Search transcript history</span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search transcripts"
        />
      </label>
      {history.length > 0 && (
        <div className="history-bulk-toolbar">
          <label className="history-select-visible">
            <input
              type="checkbox"
              checked={allVisibleSelected}
              onChange={toggleVisibleItems}
              disabled={filteredHistory.length === 0}
            />
            <CheckSquare size={14} aria-hidden="true" />
            Select visible
          </label>
          {selectedItems.length > 0 && (
            <div
              className="history-bulk-actions"
              aria-label="Bulk transcript actions"
            >
              <span>{selectedItems.length} selected</span>
              <button
                type="button"
                onClick={() =>
                  void runBulkAction("copy", () => onCopyMany(selectedItems))
                }
                disabled={Boolean(bulkAction)}
              >
                <Copy size={13} /> Copy
              </button>
              <button
                type="button"
                onClick={() =>
                  void runBulkAction("pin", () =>
                    onBulkPin(
                      selectedItems.map((item) => item.id),
                      true,
                    ),
                  )
                }
                disabled={Boolean(bulkAction)}
              >
                <Pin size={13} /> Pin
              </button>
              <button
                type="button"
                onClick={() =>
                  void runBulkAction("unpin", () =>
                    onBulkPin(
                      selectedItems.map((item) => item.id),
                      false,
                    ),
                  )
                }
                disabled={Boolean(bulkAction)}
              >
                <PinOff size={13} /> Unpin
              </button>
              <select
                value={exportFormat}
                onChange={(event) =>
                  setExportFormat(event.target.value as ExportFormat)
                }
                aria-label="History export format"
                disabled={Boolean(bulkAction)}
              >
                <option value="txt">Text</option>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
              </select>
              <button
                type="button"
                onClick={() =>
                  void runBulkAction("export", () =>
                    onExport(
                      selectedItems.map((item) => item.id),
                      exportFormat,
                    ),
                  )
                }
                disabled={Boolean(bulkAction)}
              >
                <Download size={13} /> Export
              </button>
              <button
                type="button"
                className="history-bulk-delete"
                onClick={() => {
                  if (
                    !window.confirm(
                      `Delete ${selectedItems.length} selected transcript${selectedItems.length === 1 ? "" : "s"}? This cannot be undone.`,
                    )
                  )
                    return;
                  void runBulkAction("delete", async () => {
                    const deleted = await onBulkDelete(
                      selectedItems.map((item) => item.id),
                    );
                    if (deleted) clearSelection();
                    return deleted;
                  });
                }}
                disabled={Boolean(bulkAction)}
              >
                <Trash2 size={13} /> Delete
              </button>
            </div>
          )}
        </div>
      )}
      {history.length ? (
        <div className="history-list">
          {filteredHistory.map((item) => {
            const editing = editingId === item.id;
            const selected = selectedIds.has(item.id);
            return (
              <article
                key={item.id}
                className={`${item.pinned ? "pinned" : ""}${selected ? "selected" : ""}`}
              >
                <label className="history-item-select">
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => toggleItem(item.id)}
                    aria-label={`Select transcript from ${new Date(item.createdAt).toLocaleString()}`}
                  />
                </label>
                {editing ? (
                  <textarea
                    value={editingText}
                    onChange={(event) => setEditingText(event.target.value)}
                    aria-label="Edit transcript"
                    autoFocus
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        setEditingId(null);
                        setEditingText("");
                      }
                      if (
                        event.key === "Enter" &&
                        (event.ctrlKey || event.metaKey)
                      ) {
                        event.preventDefault();
                        void saveEdit();
                      }
                    }}
                  />
                ) : (
                  <p>{item.text}</p>
                )}
                <span>
                  {item.pinned && "Pinned · "}
                  {new Date(item.createdAt).toLocaleString()} ·{" "}
                  {item.language || "Auto"}
                </span>
                <div className="history-actions">
                  {editing ? (
                    <>
                      <button type="button" onClick={() => void saveEdit()}>
                        Save
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingId(null);
                          setEditingText("");
                        }}
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => onPin(item)}
                        aria-label={
                          item.pinned ? "Unpin transcript" : "Pin transcript"
                        }
                        title={item.pinned ? "Unpin" : "Pin"}
                      >
                        {item.pinned ? <PinOff size={14} /> : <Pin size={14} />}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingId(item.id);
                          setEditingText(item.text);
                        }}
                        aria-label="Edit transcript"
                        title="Edit"
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        type="button"
                        onClick={() => onCopy(item)}
                        aria-label="Copy transcript"
                        title="Copy"
                      >
                        <Copy size={15} />
                      </button>
                    </>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      ) : settings.historyRetention === "never" ? (
        <div className="history-empty">
          <ShieldCheck size={17} /> Transcript saving is turned off.
        </div>
      ) : (
        <div className="history-empty">
          <Volume2 size={17} /> Your latest transcripts will appear here.
        </div>
      )}
      {history.length > 0 && filteredHistory.length === 0 && (
        <div className="history-empty">No transcripts match that search.</div>
      )}
    </section>
  );
}
