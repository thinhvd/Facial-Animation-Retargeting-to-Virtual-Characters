// Rig state and the blendshape generation job, shared by viewer.html and landmarks.html.
// The server decides the state (GET /api/rig/<model>, see rig_state in retarget/server.py)
// and refuses a POST the state does not allow; the pages only draw it.
window.RigJob = (() => {
  const api = m => `api/rig/${encodeURIComponent(m)}`;

  async function state(model) {
    const r = await fetch(api(model));
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  // poll while a job for `model` runs; onUpdate gets every state; resolves with the final one
  async function watch(model, onUpdate) {
    for (;;) {
      const st = await state(model);
      onUpdate(st);
      if (st.state !== 'running') return st;
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  // start a job; a refusal (busy, already up to date, ...) comes back as {refused: text}
  async function start(model, onUpdate) {
    const r = await fetch(api(model), { method: 'POST' });
    if (!r.ok) return { refused: await r.text() };
    return watch(model, onUpdate);
  }

  // progress or outcome of the last job for this model, for a small log box
  function jobText(st) {
    const j = st && st.job;
    if (!j) return '';
    const lines = j.log.map(l => l.trim()).filter(Boolean);
    switch (j.status) {
      case 'running': return `generating… ${Math.round(j.elapsed)} s (about 1 min)\n${lines.slice(-1)[0] || ''}`;
      case 'done': return `last run: done in ${j.seconds} s`;
      case 'needs_landmarks': return 'last run: MediaPipe found no face in the render';
      case 'failed': return 'last run failed:\n' + lines.slice(-6).join('\n');
      default: return '';
    }
  }

  const sourceText = lm => !lm ? '' : lm.source === 'manual' ? `${lm.count} hand-picked points` : `${lm.count} MediaPipe points`;

  return { state, watch, start, jobText, sourceText };
})();
