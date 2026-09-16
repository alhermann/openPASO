/* openPASO WebUI — Alpine.js component.
 *
 * Manages:
 *  - Model / mode / MCP-toggle UI state
 *  - Single WebSocket to /ws/{sid}, JSON event protocol
 *  - File browser + on-click visualisation
 *  - Approve/reject buttons for plan-mode tool calls
 *  - Interactive parameter sliders
 *
 * Keep this dependency-light — only Alpine + fetch + native WebSocket.
 */
// Decoded field-series data lives outside the Alpine component on purpose:
// it is megabytes of typed array and nothing about it needs to be reactive.
let _fs = null;
let _fsRaf = null;

function openpasoApp() {
  return {
    // ─── Config from backend
    models: [], mcpServers: [], modes: [],
    savedSessions: [],

    // ─── Active session state (mirrors the JSON on disk)
    session: { id: '', model: 'mock', mode: 'accept',
               mcp_servers: ['openpaso'],
               events: [], tokens_in: 0, tokens_out: 0 },

    // ─── UI state
    ws: null, connected: false, status: 'idle',
    prompt: '', events: [],
    dir: { entries: [], rel: '' }, cwd: '/',
    viz: null, params: [],
    editMode: false, saveStatus: '',
    vtkState: null, vtkRange: [0, 1],

    async init() {
      const [m, s, md] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/mcp_servers').then(r => r.json()),
        fetch('/api/modes').then(r => r.json()),
      ]);
      this.models = m.models;
      this.mcpServers = s.servers;
      this.modes = md.modes;
      this.session.model = m.default;
      this.session.mode = md.default;
      this.session.mcp_servers = s.servers
        .filter(x => x.default_on).map(x => x.id);

      await this.refreshSessionList();
      if (this.savedSessions.length > 0) {
        await this.connect(this.savedSessions[0].id);
      } else {
        await this.newSession();
      }
      this.loadFiles('');
    },

    // ─── Sessions ─────────────────────────────────────────────
    async refreshSessionList() {
      const r = await fetch('/api/sessions').then(r => r.json());
      this.savedSessions = r.sessions;
    },

    async newSession() {
      const r = await fetch('/api/sessions', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          model: this.session.model, mode: this.session.mode,
          mcp_servers: this.session.mcp_servers,
        }),
      }).then(r => r.json());
      this.savedSessions.unshift({ id: r.id, n_events: 0 });
      await this.connect(r.id);
    },

    async connect(sid) {
      if (this.ws) this.ws.close();
      this.session.id = sid;
      const res = await fetch(`/api/sessions/${sid}`);
      if (res.ok) {
        this.session = await res.json();
        this.events = this.session.events.slice();
      }
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      this.ws = new WebSocket(`${proto}://${window.location.host}/ws/${sid}`);
      this.ws.onopen = () => { this.connected = true; this.status = 'connected'; };
      this.ws.onclose = () => { this.connected = false; this.status = 'disconnected'; };
      this.ws.onerror = e => { this.status = 'error: ' + (e.message || ''); };
      this.ws.onmessage = ev => {
        try { this.onEvent(JSON.parse(ev.data)); }
        catch (e) { console.warn('bad msg', ev.data); }
      };
    },

    // ─── Inbound events ───────────────────────────────────────
    onEvent(e) {
      if (e.type === 'agent_chunk') {
        // append to last open agent_msg
        const last = this.events[this.events.length - 1];
        if (last && last.type === 'agent_msg' && last._open) {
          last.text = (last.text || '') + (e.text || '');
        } else {
          this.events.push({ type: 'agent_msg', text: e.text, _open: true });
        }
        this.scrollToBottom();
        return;
      }
      if (e.type === 'done') {
        const last = this.events[this.events.length - 1];
        if (last && last.type === 'agent_msg') last._open = false;
      }
      if (e.type === 'token_count') {
        this.session.tokens_in += e.input || 0;
        this.session.tokens_out += e.output || 0;
      }
      // A status is chrome, not a log entry. It used to be pushed like any
      // other event, which meant the connect handshake occupied the first
      // bubble (dumping the whole session object with it) and 'thinking...'
      // stayed in the scrollback forever. Both belong in the header line.
      if (e.type === 'status') {
        if (e.session) {
          this.session = Object.assign(this.session, e.session);
          this.events = this.session.events.slice();
        }
        this.status = (e.message === 'connected') ? '' : (e.message || '');
        this.scrollToBottom();
        return;
      }
      this.events.push(e);
      this.scrollToBottom();
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.log;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    // ─── Outbound ────────────────────────────────────────────
    send() {
      const text = this.prompt.trim();
      if (!text || !this.connected) return;
      this.ws.send(JSON.stringify({ type: 'prompt', text }));
      this.prompt = '';
    },

    approve(e) {
      e.decided = true;
      this.ws.send(JSON.stringify({ type: 'approve', call_id: e.call_id }));
    },
    reject(e) {
      e.decided = true;
      const reason = prompt('Reason for rejection?') || '';
      this.ws.send(JSON.stringify({ type: 'reject',
                                    call_id: e.call_id, reason }));
    },
    sendSet(type, payload) {
      if (!this.connected) return;
      this.ws.send(JSON.stringify(Object.assign({ type }, payload)));
    },
    setMode(m) {
      this.session.mode = m;
      this.sendSet('set_mode', { mode: m });
    },
    toggleMcp(id) {
      const set = new Set(this.session.mcp_servers);
      set.has(id) ? set.delete(id) : set.add(id);
      this.session.mcp_servers = [...set];
      this.sendSet('set_mcp', { servers: this.session.mcp_servers });
    },
    restart() {
      if (!confirm('Restart this session?')) return;
      this.events = [];
      this.session.tokens_in = 0; this.session.tokens_out = 0;
      this.sendSet('restart', {});
    },

    // ─── Files & viz ─────────────────────────────────────────
    async loadFiles(rel) {
      const r = await fetch('/api/files?rel=' + encodeURIComponent(rel))
        .then(r => r.json());
      this.dir = r;
      this.cwd = r.rel ? '/' + r.rel : '/';
    },

    async onClick(e) {
      this.editMode = false;
      this.saveStatus = '';
      this.viz = null;
      if (e.is_dir) {
        this.loadFiles(e.rel_path);
        return;
      }
      this.viz = { kind: 'pending', rel: e.rel_path };
      const r = await fetch('/api/viz?rel=' + encodeURIComponent(e.rel_path))
        .then(r => r.json());
      r.rel = e.rel_path;
      this.viz = r;
      if (r.kind === 'table' && r.plot) {
        this.$nextTick(() => {
          // r.plot.config carries editable/edits/toImageButtonOptions
          // from the backend so axes & titles are click-editable and
          // the toolbar exposes PNG/SVG export.
          Plotly.newPlot('plotDiv', r.plot.data, r.plot.layout,
                         r.plot.config || {responsive: true});
        });
      }
      if (r.kind === 'vtk') {
        this.$nextTick(() => this.renderVtk(r.url));
      }
      if (r.kind === 'field_series') {
        this.$nextTick(() => this.loadFieldSeries(r.url));
      }
      if (e.kind === 'text' && e.name.endsWith('.py')) {
        const pr = await fetch('/api/extract_params?rel='
                               + encodeURIComponent(e.rel_path))
          .then(r => r.json());
        this.params = pr.params;
      } else {
        this.params = [];
      }
    },

    // ─── File save (text/JSON/YAML/script editor) ───
    async saveFile() {
      if (!this.viz || !this.viz.rel || this.viz.kind !== 'text') return;
      this.saveStatus = 'saving…';
      try {
        const r = await fetch('/api/file', {
          method: 'POST',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify({rel: this.viz.rel,
                                content: this.viz.text}),
        });
        const js = await r.json();
        if (js.ok) {
          this.saveStatus = 'saved (' + js.bytes + ' B)';
          setTimeout(() => { this.saveStatus = ''; }, 4000);
          this.editMode = false;
        } else {
          this.saveStatus = 'error: ' + (js.detail || js.error || 'unknown');
        }
      } catch (e) {
        this.saveStatus = 'error: ' + e.message;
      }
    },

    // ─── Plot export ───
    exportPlot(format) {
      const div = document.getElementById('plotDiv');
      if (!div || typeof Plotly === 'undefined') return;
      const name = (this.viz && this.viz.rel)
        ? this.viz.rel.split('/').pop().replace(/\.\w+$/, '') : 'plot';
      Plotly.downloadImage(div, {format, filename: name, scale: 2});
    },

    // ─── VTK rendering (real, vtk.js HTTPDataAccessHelper) ───
    // ── Field series ────────────────────────────────────────────
    // A solver's own field, sampled onto a grid once per stored timestep and
    // played back. Signed quantities get a diverging ramp with the canvas at
    // zero, so the sign is the thing you see first.
    fsPlaying: true,
    fsFrame: 0,
    fsNFrames: 0,
    fsTime: 0,

    fsColormap() {
      // coral for one sign, light graphit for the other, canvas at zero.
      const lut = new Uint8Array(256 * 3);
      const neg = [0xB6, 0xC2, 0xD2], zero = [0x0D, 0x11, 0x17], pos = [0xFF, 0x6B, 0x4A];
      for (let i = 0; i < 256; i++) {
        const t = (i / 255) * 2 - 1;            // -1 .. +1
        const a = Math.abs(t);
        const end = t < 0 ? neg : pos;
        // ease so the quiet field stays dark and structure reads early
        const w = Math.pow(a, 0.65);
        for (let c = 0; c < 3; c++) {
          lut[i * 3 + c] = Math.round(zero[c] + (end[c] - zero[c]) * w);
        }
      }
      return lut;
    },

    async loadFieldSeries(url) {
      const canvas = document.getElementById('fieldCanvas');
      if (!canvas) return;
      if (_fsRaf) { cancelAnimationFrame(_fsRaf); _fsRaf = null; }
      try {
        const meta = await fetch(url).then(r => r.json());
        const b2a = (b64) => {
          const bin = atob(b64);
          const out = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
          return out;
        };
        _fs = {
          nx: meta.nx, ny: meta.ny,
          times: meta.times || [],
          fps: meta.fps || 25,
          mask: b2a(meta.mask),
          frames: b2a(meta.frames),
          lut: this.fsColormap(),
        };
        _fs.n = _fs.times.length;
        canvas.width = _fs.nx; canvas.height = _fs.ny;
        this.fsNFrames = _fs.n;
        this.fsFrame = 0;
        this.fsPlaying = true;
        this.fsLoop(canvas);
      } catch (e) {
        this.viz = { kind: 'error', error: 'field series: ' + this.scrub(e.message) };
      }
    },

    fsDraw(canvas) {
      if (!_fs) return;
      const ctx = canvas.getContext('2d');
      const { nx, ny, mask, frames, lut } = _fs;
      const img = ctx.createImageData(nx, ny);
      const base = this.fsFrame * nx * ny;
      for (let row = 0; row < ny; row++) {
        // the grid's first row is the bottom of the domain; canvas y runs down
        const src = (ny - 1 - row) * nx;
        for (let col = 0; col < nx; col++) {
          const s = src + col, d = (row * nx + col) * 4;
          if (!mask[s]) {                       // inside the obstacle
            img.data[d] = 0x21; img.data[d+1] = 0x26; img.data[d+2] = 0x2D;
            img.data[d+3] = 255;
            continue;
          }
          const v = frames[base + s] * 3;
          img.data[d] = lut[v]; img.data[d+1] = lut[v+1]; img.data[d+2] = lut[v+2];
          img.data[d+3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      this.fsTime = _fs.times[this.fsFrame] || 0;
    },

    fsLoop(canvas) {
      let last = 0;
      const tick = (now) => {
        if (!_fs) return;
        const interval = 1000 / _fs.fps;
        if (this.fsPlaying && now - last >= interval) {
          this.fsFrame = (this.fsFrame + 1) % _fs.n;
          this.fsDraw(canvas);
          last = now;
        }
        _fsRaf = requestAnimationFrame(tick);
      };
      this.fsDraw(canvas);
      // A viewer who asked for less motion gets the first frame and a scrubber.
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        this.fsPlaying = false;
      }
      _fsRaf = requestAnimationFrame(tick);
    },

    fsSeek(i) {
      if (!_fs) return;
      this.fsFrame = Math.max(0, Math.min(_fs.n - 1, parseInt(i, 10) || 0));
      const c = document.getElementById('fieldCanvas');
      if (c) this.fsDraw(c);
    },

    async renderVtk(url) {
      const root = document.getElementById('vtkRoot');
      if (!root || typeof vtk === 'undefined') {
        if (root) root.innerHTML = '<div class="text-slate-500 p-2">' +
          'vtk.js library failed to load from CDN; check network.</div>';
        return;
      }
      root.innerHTML = '';
      const fsContainer = vtk.Rendering.Misc.vtkFullScreenRenderWindow
        ? vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
            rootContainer: root, background: [0.051, 0.067, 0.090],  // #0D1117
          }) : null;
      if (!fsContainer) return;
      const renderer = fsContainer.getRenderer();
      const renderWindow = fsContainer.getRenderWindow();
      const reader = vtk.IO.XML.vtkXMLUnstructuredGridReader
        ? vtk.IO.XML.vtkXMLUnstructuredGridReader.newInstance() : null;
      if (!reader) {
        root.innerHTML = '<div class="text-slate-500 p-2">' +
          'vtk.js modules missing. Load the full vtk.js bundle.</div>';
        return;
      }
      // The guard has to cover the render too, not just the fetch: an
      // unreadable grid throws at getOutputData, well past the old catch.
      try {
        const resp = await fetch(url);
        const buf = await resp.arrayBuffer();
        reader.parseAsArrayBuffer(buf);

        const grid = reader.getOutputData(0);
        if (!grid || !grid.getNumberOfPoints || grid.getNumberOfPoints() === 0) {
          throw new Error('the file parsed but carries no points');
        }
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputData(grid);
        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);
        renderer.addActor(actor);
        renderer.resetCamera();
        renderWindow.render();
        this.vtkState = { fsContainer, renderer, renderWindow, mapper, actor };

        const arr = grid.getPointData().getScalars();
        if (arr) {
          const r = arr.getRange();
          this.vtkRange = [r[0], r[1]];
        }
      } catch (e) {
        root.innerHTML = '<div class="text-slate-300 p-2">' +
          'This file could not be displayed: ' + this.scrub(e.message) + '</div>';
      }
    },

    resetVtkCamera() {
      if (!this.vtkState) return;
      this.vtkState.renderer.resetCamera();
      this.vtkState.renderWindow.render();
    },
    applyVtkRange() {
      if (!this.vtkState) return;
      const m = this.vtkState.mapper;
      if (m && m.setScalarRange) {
        m.setScalarRange(this.vtkRange[0], this.vtkRange[1]);
        this.vtkState.renderWindow.render();
      }
    },
    saveVtkScreenshot() {
      if (!this.vtkState) return;
      const canvas = this.vtkState.fsContainer.getOpenGLRenderWindow()
        .getCanvas();
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = (this.viz && this.viz.name) ? this.viz.name + '.png'
        : 'render.png';
      a.click();
    },

    rerunWithParams() {
      const summary = this.params
        .map(p => `${p.name} = ${p.value}`).join('; ');
      this.prompt = 'Re-run the previous script with these parameter ' +
        'changes and report the result file: ' + summary;
      this.send();
    },

    // ─── Helpers ─────────────────────────────────────────────
    bubbleClass(e) {
      const m = {
        user_msg:           'border-accent-500/30 bg-accent-500/8 text-slate-100 ml-auto',
        agent_msg:          'border-ink-700 bg-ink-800/70 text-slate-100',
        agent_chunk:        'border-ink-700 bg-ink-800/70 text-slate-100',
        tool_call_pending:  'border-state-call/40 bg-state-call/8 text-slate-100',
        tool_call_executing:'border-state-call/40 bg-state-call/12 text-slate-100',
        tool_result:        'border-ink-700 bg-ink-900/80 text-slate-300',
        subagent_spawned:   'border-state-sub/40 bg-state-sub/10 text-slate-100',
        subagent_returned:  'border-state-sub/40 bg-state-sub/8 text-slate-200',
        token_count:        'border-ink-700/50 bg-ink-900/40 text-slate-500 text-[10px] py-1.5',
        error:              'border-state-err/40 bg-state-err/12 text-slate-100',
        done:               'border-accent-500/30 bg-accent-500/8 text-accent-300',
        status:             'border-ink-700/50 bg-ink-900/40 text-slate-500 text-[10px] py-1.5',
      };
      return m[e.type] || 'border-ink-700 bg-ink-800/50 text-slate-200';
    },
    eventIcon(e) {
      return ({
        user_msg: '👤', agent_msg: '🤖', agent_chunk: '🤖',
        tool_call_pending: '⚙', tool_call_executing: '⚙',
        tool_result: '✓',
        subagent_spawned: '👁', subagent_returned: '✓',
        token_count: '∑', error: '⚠', done: '●', status: '·',
        tool_call_rejected: '⊘', tool_error: '⚠',
      })[e.type] || '·';
    },
    eventLabel(e) {
      const m = {
        user_msg: 'You', agent_msg: 'Agent', agent_chunk: 'Agent',
        tool_call_pending: 'Tool call (pending)',
        tool_call_executing: 'Tool call',
        tool_result: 'Tool result',
        subagent_spawned: 'Sub-agent',
        subagent_returned: 'Sub-agent return',
        token_count: 'Tokens',
        error: 'Error', done: 'Done', status: 'Status',
        tool_call_rejected: 'Tool call rejected',
        tool_error: 'Tool failed',
      };
      return m[e.type] || e.type;
    },
    eventClass(e) { return this.bubbleClass(e); },
    // Never print a machine's home directory on someone else's screen.
    scrub(s) {
      return String(s == null ? '' : s).replace(/\/home\/[^/\s:'"]+\//g, '~/');
    },
    formatEvent(e) {
      if (e.type === 'agent_msg' || e.type === 'user_msg') return e.text || '';
      if (e.type === 'tool_call_pending' || e.type === 'tool_call_executing') {
        return `${e.tool} (${JSON.stringify(e.args || {}).slice(0, 200)})`;
      }
      if (e.type === 'tool_result') return (e.result || '').slice(0, 800);
      if (e.type === 'subagent_spawned') {
        return `${e.role}: ${e.task}\n[context] ${e.context || ''}`;
      }
      if (e.type === 'subagent_returned') return (e.result || '').slice(0, 800);
      if (e.type === 'token_count') return `in=${e.input}  out=${e.output}`;
      // These three used to fall through to JSON.stringify and render as
      // {"message":"thinking…"} on screen, handshake payload and all.
      if (e.type === 'status') {
        return e.message === 'connected' ? 'Connected.' : this.scrub(e.message || '');
      }
      if (e.type === 'done') return '';
      if (e.type === 'error' || e.type === 'tool_error') {
        return this.scrub(e.message || e.error || '').slice(0, 800);
      }
      if (e.type === 'tool_call_rejected') {
        return this.scrub(`${e.tool || 'tool'}${e.reason ? ': ' + e.reason : ''}`);
      }
      const { type, ...rest } = e;
      return this.scrub(JSON.stringify(rest)).slice(0, 400);
    },
    kindIcon(k) {
      return { dir: '📁', vtk: '🌐', hdf: '📦', image: '🖼',
               table: '📊', json: '{ }', yaml: '⚙', mesh: '🔲',
               text: '📄', binary: '⬛' }[k] || '·';
    },
    humanSize(n) {
      if (n == null) return '';
      if (n < 1024) return n + ' B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
      return (n / 1024 / 1024).toFixed(1) + ' MB';
    },
  };
}
