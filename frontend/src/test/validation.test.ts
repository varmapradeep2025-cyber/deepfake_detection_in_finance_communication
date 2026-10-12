import { describe, expect, it } from "vitest";
import { validateFile, validateUrl } from "../lib/validation";

function makeFile(name: string, type: string, size = 1024): File {
  return new File([new Uint8Array(size)], name, { type });
}

describe("validateFile", () => {
  it("accepts a supported image extension and size", () => {
    expect(validateFile(makeFile("photo.png", "image/png"), "image").valid).toBe(
      true,
    );
  });

  it("rejects an unsupported extension", () => {
    const result = validateFile(makeFile("script.exe", "application/octet-stream"), "image");
    expect(result.valid).toBe(false);
    expect(result.error).toMatch(/Unsupported extension/);
  });

  it("rejects an empty file", () => {
    const result = validateFile(makeFile("empty.png", "image/png", 0), "image");
    expect(result.valid).toBe(false);
    expect(result.error).toMatch(/empty/);
  });

  it("rejects a file above the modality limit", () => {
    const oversized = makeFile("big.png", "image/png", 30 * 1024 * 1024);
    const result = validateFile(oversized, "image");
    expect(result.valid).toBe(false);
    expect(result.error).toMatch(/limit/);
  });

  it("rejects file input for the URL modality", () => {
    const result = validateFile(makeFile("x.png", "image/png"), "url");
    expect(result.valid).toBe(false);
  });
});

describe("validateUrl", () => {
  it("accepts an https URL", () => {
    expect(validateUrl("https://example.com/login").valid).toBe(true);
  });

  it("accepts an http URL", () => {
    expect(validateUrl("http://example.com").valid).toBe(true);
  });

  it("rejects an empty string", () => {
    expect(validateUrl("   ").valid).toBe(false);
  });

  it("rejects a non-http scheme", () => {
    const result = validateUrl("ftp://example.com");
    expect(result.valid).toBe(false);
    expect(result.error).toMatch(/scheme/);
  });

  it("rejects a URL with spaces", () => {
    expect(validateUrl("https://exa mple.com").valid).toBe(false);
  });

  it("rejects a hostname without a dot", () => {
    const result = validateUrl("https://localhost");
    expect(result.valid).toBe(false);
    expect(result.error).toMatch(/domain/);
  });

  it("rejects a malformed URL", () => {
    expect(validateUrl("not a url").valid).toBe(false);
  });
});
