import { Link2 } from "lucide-react";

interface UrlInputProps {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}

export function UrlInput({ value, disabled, onChange }: UrlInputProps) {
  return (
    <label className="field">
      <span className="field__label">
        <Link2 size={14} aria-hidden /> URL to inspect
      </span>
      <input
        type="url"
        className="input"
        placeholder="https://example.com/payment/redirect"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        aria-label="URL to inspect"
        spellCheck={false}
      />
      <span className="field__hint">
        No request is sent. The URL is checked for syntax and scheme only; it is
        not fetched, resolved, or reputation-scored.
      </span>
    </label>
  );
}
