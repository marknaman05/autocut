// A <video> stand-in whose seek events fire *after* the current JS turn, the
// way a real element dispatches them.  Firing them synchronously hides
// exactly the class of bug this exists to catch.
export class FakeVideo {
  constructor() {
    this.dataset = {}; this.listeners = {}; this.playing = false;
    this.seeks = []; this._t = 0; this._pending = [];
  }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  fire(name) { (this.listeners[name] || []).forEach(fn => fn()); }
  get currentTime() { return this._t; }
  set currentTime(v) { this._t = v; this.seeks.push(v); this._pending.push("seeking", "seeked"); }
  play() { this.playing = true; }
  pause() { this.playing = false; }
  /** Deliver the seek events queued by the caller's turn. */
  flush() { const q = this._pending; this._pending = []; q.forEach(n => this.fire(n)); }
  advance(to, step = 0.05) {
    this.flush();
    while (this._t < to && this.playing) {
      this._t = Math.min(this._t + step, to);
      this.fire("timeupdate");
      this.flush();
    }
  }
}
