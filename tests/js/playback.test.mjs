// The review page's playback state machine, driven against a fake <video>.
//
// This lives in JS rather than Python because the logic under test *is* the
// browser's event ordering: a seek dispatches its events after the current
// turn, and code that assumes otherwise breaks in a way no amount of reading
// catches.  Run by tests/test_playback.py, or directly with
// `node tests/js/playback.test.mjs`.
import { readFileSync } from "node:fs";
import assert from "node:assert";
import { FakeVideo } from "./fakevideo.mjs";

const here = new URL(".", import.meta.url).pathname;
const html = readFileSync(`${here}/../../server/static/index.html`, "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const body = script.slice(script.indexOf("function playSpans"));
const MIN_PREVIEW = 0.05;
const { playSpans } = new Function(
  "MIN_PREVIEW", body + "\nreturn { playSpans };"
)(MIN_PREVIEW);

let failures = 0;
const check = (name, fn) => {
  try { fn(); console.log(`  ok   ${name}`); }
  catch (e) { failures++; console.log(`  FAIL ${name}\n       ${e.message}`); }
};

check("hearing one part plays it and stops at its end", () => {
  const v = new FakeVideo();
  playSpans(v, [[36.9, 43.9]]);
  assert.strictEqual(v.seeks[0], 36.9);
  v.advance(43.9);
  assert.ok(!v.playing, "must stop at the end of the part");
});

check("REGRESSION: a second click also stops", () => {
  const v = new FakeVideo();
  playSpans(v, [[36.9, 43.9]]);
  v.advance(44.0);
  // The bug: the seek fired 'seeking' after stopAt was set, cancelling the
  // plan, so every click after the first played to the end of the video.
  playSpans(v, [[36.9, 43.9]]);
  assert.ok(v.playing, "second click should start playing");
  v.advance(65.5);
  assert.ok(!v.playing, "second click must stop too");
  assert.ok(v.currentTime < 45, `ran on to ${v.currentTime}`);
});

check("playing the edit stitches kept parts and skips the rest", () => {
  const v = new FakeVideo();
  // Three kept parts with dropped material between them.
  playSpans(v, [[1.35, 9.94], [11.51, 15.91], [19.03, 23.18]]);
  assert.strictEqual(v.seeks[0], 1.35);
  v.advance(9.94);
  assert.strictEqual(v.seeks[1], 11.51, "jumps over the dropped part");
  v.advance(15.91);
  assert.strictEqual(v.seeks[2], 19.03, "jumps over the second dropped part");
  assert.ok(v.playing, "still playing through the stitch");
  v.advance(23.18);
  assert.ok(!v.playing, "stops after the last kept part");
});

check("adjacent kept parts play without a gap", () => {
  const v = new FakeVideo();
  playSpans(v, [[0, 5], [5, 10]]);
  v.advance(5);
  assert.strictEqual(v.seeks[1], 5, "seeks to the same instant, so no audio is lost");
  v.advance(10);
  assert.ok(!v.playing);
});

check("scrubbing hands control back to the viewer", () => {
  const v = new FakeVideo();
  playSpans(v, [[36.9, 43.9]]);
  v.currentTime = 10;          // the viewer drags the scrubber
  v.playing = true;
  v.advance(60);
  assert.ok(v.playing, "must not be steered or paused after a manual seek");
});

check("a negative start is clamped to zero", () => {
  const v = new FakeVideo();
  playSpans(v, [[-1.2, 1.35]]);
  assert.ok(v.seeks.every(t => t >= 0), `negative seek: ${v.seeks}`);
});

check("a zero-length span is skipped, not played", () => {
  const v = new FakeVideo();
  playSpans(v, [[5, 5], [10, 12]]);
  assert.strictEqual(v.seeks[0], 10, "starts at the first span with any length");
});

check("an empty selection plays nothing", () => {
  const v = new FakeVideo();
  playSpans(v, []);
  assert.ok(!v.playing, "must not start playing with nothing selected");
});

console.log(failures ? `\n${failures} failing` : "\nall passing");
process.exit(failures ? 1 : 0);
