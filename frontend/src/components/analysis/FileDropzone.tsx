import { useRef, useState, type DragEvent } from "react";
import { FileUp, Trash2, UploadCloud } from "lucide-react";
import type { Modality } from "../../types";
import { formatBytes } from "../../lib/format";
import { getModalityMeta } from "../../lib/validation";

interface FileDropzoneProps {
  modality: Modality;
  file: File | null;
  disabled?: boolean;
  onFile: (file: File) => void;
  onClear: () => void;
}

export function FileDropzone({
  modality,
  file,
  disabled,
  onFile,
  onClear,
}: FileDropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const meta = getModalityMeta(modality);

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!disabled) setDragging(true);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFile(dropped);
  }

  if (file) {
    return (
      <div className="file-card" data-testid="selected-file">
        <span className="file-card__icon" aria-hidden>
          <FileUp size={20} />
        </span>
        <div className="file-card__meta">
          <span className="file-card__name" title={file.name}>
            {file.name}
          </span>
          <span className="file-card__details">
            {formatBytes(file.size)} · {meta.label} · {file.type || "unknown MIME"}
          </span>
        </div>
        <button
          type="button"
          className="btn btn--ghost btn--sm"
          onClick={onClear}
          disabled={disabled}
        >
          <Trash2 size={14} aria-hidden />
          Remove
        </button>
      </div>
    );
  }

  return (
    <div
      className={`dropzone${dragging ? " dropzone--active" : ""}${
        disabled ? " dropzone--disabled" : ""
      }`}
      onDragOver={handleDragOver}
      onDragEnter={handleDragOver}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <UploadCloud size={26} aria-hidden className="dropzone__glyph" />
      <p className="dropzone__title">
        Drag and drop a {meta.label.toLowerCase()} file here
      </p>
      <p className="dropzone__hint">
        {meta.extensions.join(", ")} · up to{" "}
        {formatBytes(meta.maxBytes)} · never uploaded to an external service
      </p>
      <button
        type="button"
        className="btn btn--primary btn--sm"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
      >
        Choose file
      </button>
      <input
        ref={inputRef}
        type="file"
        className="visually-hidden"
        accept={meta.accept.join(",")}
        aria-label={`Select ${meta.label.toLowerCase()} file`}
        onChange={(event) => {
          const selected = event.target.files?.[0];
          if (selected) onFile(selected);
          event.target.value = "";
        }}
        disabled={disabled}
      />
    </div>
  );
}
