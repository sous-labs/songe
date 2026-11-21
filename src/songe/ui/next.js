// Learning cycle page (draft): renders the `cycle` section of /api/state
// (docs/CONTRACTS.md §8, Learning cycle) with words from wording.json.
const $ = id => document.getElementById(id);
let W = null;
let busy = false;
const open = new Set();          // details the viewer opened, kept across renders
const fullRecords = new Map();   // cycle id → full record, once asked for
const loadingFull = new Map();   // cycle id → 'loading' | 'failed' while asked for

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    // Handlers and styles go through the DOM: the page's content security
    // policy blocks inline handler and style attributes.
    if (k === 'class') node.className = v;
    else if (typeof v === 'function' && k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (k === 'style') node.style.cssText = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) if (c != null && c !== false) node.append(c instanceof Node ? c : String(c));
  return node;
}
const plural = (n, unit) => {
  const [one, many] = W.unit[unit] || [unit, unit + 's'];
  return `${n} ${n === 1 ? one : many}`;
};
const fill = (text, values) => text.replace(/\{(\w+)\}/g, (_, k) => values[k] ?? '');
function duration(s) {
  if (s == null) return '';
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${Math.floor(s / 3600)}h ${Math.round(s % 3600 / 60)}m`;
}
function value(unit, v) {
  if (v == null) return '–';
  if (unit === 'accuracy' || unit === 'share') return `${Math.round(v * 100)}%`;
  return v.toFixed(4);
}
function short(text, n) {
  text = String(text || '').replace(/\s+/g, ' ').trim();
  return text.length > n ? text.slice(0, n - 1) + '…' : text;
}
// details that remember being opened across re-renders
function remembered(key, summary, body, initiallyOpen = false) {
  const node = el('details', {class: 'stage-wrap'}, el('summary', {}, summary), body);
  node.open = open.has(key) || (initiallyOpen && !open.has('closed:' + key));
  node.addEventListener('toggle', () => {
    if (node.open) { open.add(key); open.delete('closed:' + key); } else { open.delete(key); open.add('closed:' + key); }
  });
  return node;
}

// One line in the rail saying what a control did, or why it could not.
// What a control did, or why it could not: shown under the model's header,
// next to the control that was used. The page redraws every poll, so the
// notice lives here and each render draws it until it expires or is closed.
let notice = null;            // {text, failure, until}
let dreamMenu = null;         // model id whose Dream menu is open
document.addEventListener('click', e => {
  if (dreamMenu && !e.target.closest('.dream-wrap')) { dreamMenu = null; render(); }
});
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if (dreamMenu) { dreamMenu = null; render(); }
});
let noticeTimer = null;
function say(text, failure = false, action = null) {
  notice = {text, failure, action, until: Date.now() + (failure ? 20000 : 15000)};
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice = null; render(); }, failure ? 20000 : 15000);
  render();
}
function noticeView() {
  if (!notice || Date.now() > notice.until) return null;
  return el('div', {class: 'notice', role: 'status', 'data-failure': String(notice.failure)},
    el('p', {}, notice.text, notice.action ? [' · ', el('button', {type: 'button', class: 'notice-action',
      onclick: () => { const run = notice.action.run; notice = null; run(); }}, notice.action.label)] : null),
    el('button', {type: 'button', class: 'notice-close', 'aria-label': 'Close',
      onclick: () => { notice = null; clearTimeout(noticeTimer); render(); }}, '×'));
}
async function control(action, payload) {
  if (busy) return;
  busy = true;
  try {
    const r = await fetch('/api/control', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({action, ...payload})});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `request failed (${r.status})`);
    if (action === 'undo') say(W.restored);
    else if (action === 'pause' || action === 'resume') say(W.dream_notice[action]);
    else if (data.since) say(fill(W.quick_nothing, {when: when(data.since)}));  // nothing new since then
    else if (data.message) say(data.message);  // Dream: what started, or what it waits on
  } catch (e) {
    // No answer at all (a restarted or replaced server): not a Songe error.
    say(e instanceof TypeError ? W.control_unreachable : fill(W.control_failed, {error: e.message}), true);
  } finally { busy = false; refresh(); }
}

// ---- the three status lines ----
function statusLines(models, inventory, paused) {
  const name = id => inventory[id]?.name || id;
  const activeId = Object.keys(models).find(id => models[id].current);
  const active = activeId ? models[activeId] : null;
  const lastId = Object.keys(models).filter(id => models[id].last).sort((a, b) => (models[b].last.ended_at || 0) - (models[a].last.ended_at || 0))[0];
  const last = lastId ? models[lastId] : null;
  if (active) {
    const c = active.current;
    const i = c.stages.findIndex(s => s.key === c.stage);
    const stage = c.stages[i];
    const p = stage.progress;
    const now = p?.unit === 'calibration' ? fill(W.getting_ready_title, p)
      : [W.stage[c.stage], p && (p.total ? `${p.done} of ${plural(p.total, p.unit)}` : plural(p.done, p.unit)), stage.state === 'now' && stage.eta_seconds != null && `about ${duration(stage.eta_seconds)} left`].filter(Boolean).join(' · ');
    const following = c.stages.slice(i + 1).find(s => s.state === 'ahead');
    const next = following && following.key !== 'result' ? `${W.stage[following.key]} · after ${W.stage[c.stage].toLowerCase()}`
      : 'Result · uses the new version only if it passes its checks';
    return [`${name(activeId)} · ${now}`, c.wait ? W.wait[c.wait.code] : next];
  }
  // Measuring this machine (not a Dream).
  const measuringId = Object.keys(models).find(id => models[id].measuring_machine);
  if (measuringId) {
    const m = models[measuringId].measuring_machine;
    return [`${name(measuringId)} · ${W.measuring_short}${m.total ? ` · ${fill(W.measuring_progress, m)}` : ''}`, W.measuring_next];
  }
  // Reading (in the background, or a Quick Dream): say that, and what follows.
  const readingId = Object.keys(models).find(id => models[id].quick_running);
  if (readingId) {
    const q = models[readingId].quick_running;
    const now = [W.quick_title[q.kind] || W.quick_title.quick,
      q.stage === 'read' && q.total ? `${q.done} of ${plural(q.total, 'conversations_read')}` : null,
      q.stage === 'read' && q.eta_seconds != null ? `about ${duration(q.eta_seconds)} left` : null].filter(Boolean).join(' · ');
    return [`${name(readingId)} · ${now}`, q.kind === 'quick' ? W.next_after_quick : W.next_after_reading];
  }
  // What the next Deep Dream waits on, when it is conversations to check with.
  const short = Object.values(models).map(m => m.collecting).find(Boolean);
  const needs = short && fill(W.collecting_next, {count: short.need - short.have, s: short.need - short.have === 1 ? '' : 's'});
  if (last) {
    const r = last.last.result;
    const auto = inventory[lastId]?.auto_dream ?? true;
    return [`Idle · ${r?.outcome === 'adopted' ? `${name(lastId)} moved to ${r.version}` : `last Deep Dream not used`} ${when(last.last.ended_at)}`,
      needs || (auto ? 'Dreams again when your computer is idle' : 'Auto-dream is off · press Dream to run one')];
  }
  const firstId = Object.keys(models)[0];
  const pending = models[firstId]?.pending;
  const auto = inventory[firstId]?.auto_dream ?? true;
  return [`Collecting · ${pending ? plural(pending.conversations, 'conversations') : ''} recorded`,
    auto ? 'Press Dream to learn now, or it starts when idle' : 'Auto-dream is off · press Dream to run one'];
}

// ---- before/after ----
// Long absolute paths say little; keep the last two parts.
// What the user would see of an answer: no private reasoning, no tool-call
// markup written as text (the call is shown parsed above it).
function visible(t) {
  let v = String(t || '').replace(/<think>[\s\S]*?<\/think>/g, '');
  if (v.includes('</think>')) v = v.slice(v.lastIndexOf('</think>') + 8);
  return v.split(/<tool_call>|<\|tool_call|<function=/)[0].trim();
}
const tidy = v => String(v).replace(/(?:\/[\w.@-]+){3,}/g, m => '…/' + m.split('/').slice(-2).join('/'));
function callText(answer) {
  const c = answer.tool_calls?.[0];
  if (!c) return '';
  const body = callBody(answer);
  const args = Object.values(c.arguments || {}).filter(v => v !== body)
    .map(v => short(tidy(typeof v === 'string' ? v : JSON.stringify(v)), 60));
  return `${c.name}(${args.join(', ')})`;
}
// What a call writes or runs (a file's content, a command): its longest text
// argument, shown in full below the call, since that is where a preference shows.
function callBody(answer) {
  const c = answer.tool_calls?.[0];
  const texts = Object.values(c?.arguments || {}).filter(v => typeof v === 'string' && v.length > 60);
  return texts.sort((a, b) => b.length - a.length)[0] || '';
}
// Word-level diff: which tokens of `a` are not in `b`'s longest common run.
function changedTokens(a, b) {
  const x = a.split(/(\s+)/), y = b.split(/(\s+)/);
  if (x.length * y.length > 400000) return x.map(t => [t, false]);
  const n = x.length, m = y.length, L = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n) {
    if (j < m && x[i] === y[j]) { out.push([x[i], false]); i++; j++; }
    else if (j < m && L[i][j + 1] >= L[i + 1][j]) j++;
    else { out.push([x[i], !/^\s+$/.test(x[i])]); i++; }
  }
  return out;
}
// Join changed words (and the spaces between them) into one mark per run.
function marked(text, other, kind) {
  const tokens = changedTokens(text, other), out = [];
  let run = null;
  tokens.forEach(([t, changed], i) => {
    const space = /^\s+$/.test(t);
    const bridges = space && run && tokens[i + 1]?.[1];
    if (changed || bridges) { if (!run) out.push(run = el('mark', {class: kind})); run.append(t); }
    else { run = null; out.push(t); }
  });
  return out;
}
function answerBox(kind, answer, other, key) {
  const text = tidy(visible(answer.text)), otherText = tidy(visible(other.text));
  const call = callText(answer), otherCall = callText(other);
  const body = callBody(answer), otherBody = callBody(other);
  const more = el('button', {type: 'button', class: 'more'}, open.has(key) ? 'Show less' : 'Show all');
  const box = el('div', {class: `answer ${kind}`, 'data-open': String(open.has(key))},
    el('h4', {}, W[kind]),
    el('div', {class: call ? 'call' : 'call none'}, call ? marked(call, otherCall, kind) : 'No tool call'),
    body ? el('pre', {class: 'call-body'}, marked(body, otherBody, kind)) : null,
    text ? el('div', {class: 'prose'}, marked(text, otherText, kind)) : null,
    text.length > 280 || body.split('\n').length > 8 ? more : null);
  more.onclick = () => {
    if (open.has(key)) open.delete(key); else open.add(key);
    box.dataset.open = String(open.has(key));
    more.textContent = open.has(key) ? 'Show less' : 'Show all';
  };
  return box;
}
function example(e) {
  const from = e.source === 'original' ? W.example_source.original : W.example_source.practice;
  const said = e.signal.source === 'user' ? 'after you said' : `after ${W.signal_type[e.signal.type]}`;
  return el('article', {class: 'example'},
    // What to look for first: the lesson this moment is about.
    e.lesson ? el('p', {class: 'example-lesson'}, el('span', {class: 'muted'}, `${W.example_lesson} · `), short(e.lesson, 160)) : null,
    el('div', {class: 'pair'},
      answerBox('before', e.before, e.after, `${e.id}:before`),
      answerBox('after', e.after, e.before, `${e.id}:after`)),
    el('p', {class: 'example-from'}, from,
      e.feedback ? [` · ${said} `, el('q', {}, short(e.feedback, 120))] : null),
    el('p', {class: 'example-task'}, short(tidy(e.context.request), 140)));
}
function when(sec) {
  if (sec == null) return '';
  const d = new Date(sec * 1000), today = new Date();
  const time = d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  return d.toDateString() === today.toDateString() ? `today ${time}` : `${d.toLocaleDateString([], {month: 'short', day: 'numeric'})} ${time}`;
}
// The result in words. After an undo, an adopted version was used then,
// not now.
function outcomeText(r) {
  const undone = r.outcome === 'adopted' && r.in_use === false;
  return fill(undone ? W.adopted_then : W.outcome[r.outcome], {version: r.version});
}
function finished(c, current) {
  const r = c.result;
  const done = r.outcome === 'adopted';
  const by = Object.fromEntries(c.stages.map(s => [s.key, s.progress]));
  const did = [
    by.read && `read ${plural(by.read.total, 'conversations')}`,
    // Practice counts scenes written (no total): the same words as its row.
    c.practice?.written && `${c.practice.kept} of ${plural(c.practice.written, 'scenes')} kept`,
    c.train.total && `trained ${plural(c.train.total, 'steps')}`,
    by.check && `ran ${plural(by.check.done, 'checks')}`,
  ].filter(Boolean).join(' · ');
  return el('div', {class: `finished ${done ? 'adopted' : 'kept'}`},
    el('p', {class: 'finished-title'}, el('span', {class: 'tick'}, '✓'), ' Dream finished',
      el('span', {class: 'muted'}, ` · ${when(c.ended_at)} · took ${duration(c.ended_at - c.started_at)}`)),
    // After an undo it was adopted then, and is not what runs now.
    el('p', {class: 'finished-outcome'}, outcomeText(r), el('span', {class: 'muted'}, ` · ${W.reason[r.reason]}`)),
    done && r.in_use === false ? el('p', {class: 'finished-outcome muted'}, fill(W.undone, {version: r.version, current: current?.label || 'Original'})) : null,
    el('p', {class: 'muted'}, did));
}
// The same spot as "Dream finished", while a Dream runs: which stage, how far.
function dreaming(c) {
  const i = c.stages.findIndex(s => s.key === c.stage);
  const stage = c.stages[i], p = stage.progress;
  const now = c.stage === 'measure' || p?.unit === 'calibration' ? fill(W.getting_ready_title, p || {done: 0, total: '?'})
    : [W.stage[c.stage], p && p.total && `${p.done} of ${plural(p.total, p.unit)}`,
      stage.state === 'now' && stage.eta_seconds != null && `about ${duration(stage.eta_seconds)} left`].filter(Boolean).join(' · ');
  const steps = el('div', {class: 'steps', 'aria-hidden': 'true', style: `grid-template-columns:repeat(${c.stages.length},1fr)`}, c.stages.map(s => {
    const share = s.state === 'done' ? 1 : s.state === 'now' || s.state === 'waiting' ? (s.progress?.total ? s.progress.done / s.progress.total : 0) : 0;
    return el('span', {title: W.stage[s.key]}, el('i', {style: `width:${Math.round(share * 100)}%`}));
  }));
  return el('div', {class: `finished running${c.wait ? ' waiting' : ''}`},
    el('p', {class: 'finished-title'}, el('span', {class: 'tick'}, c.wait ? '‖' : '●'), c.wait ? ' Dreaming · paused' : ' Dreaming',
      el('span', {class: 'muted'}, ` · started ${when(c.started_at)} · step ${i + 1} of ${c.stages.length}`)),
    el('p', {class: 'finished-outcome'}, c.wait ? W.wait[c.wait.code] : now),
    steps,
    el('p', {class: 'muted'}, 'Keeps the current version unless the new one passes its checks'),
    el('p', {class: 'dream-actions'},
      // Pause only suspends a Dream; Cancel throws this one away.
      el('button', {type: 'button', class: 'text-action', onclick: () => {
        if (window.confirm(W.cancel_confirm)) control('cancel', {model: c.model});
      }}, W.cancel_dream)));
}
// What a Deep Dream changed: its before/after examples, or what its checks
// decided on when no answer is clearly better.
// Judged gates (better / worse / same at each moment) or the older count.
function improvedText(i) {
  return i.worse != null ? fill(W.judged, i) : fill(W.improved, i);
}
function whatChanged(c) {
  const r = c.result;
  const section = el('section', {class: 'changed'},
    el('h3', {class: 'section-title'}, 'What changed', r.improved?.count != null ? el('span', {class: 'muted'}, ` · ${improvedText(r.improved)}`) : null));
  if (!r.examples.length) {
    // Never judged (made before the check existed): say so, not "none improved".
    section.append(el('p', {class: 'empty'}, r.improved?.count == null && r.improved?.available ? W.examples_unchecked : W.nothing_improved));
    const h = r.held_out;
    if (h && h.change != null && h.margin != null) {
      const n = v => (v >= 0 ? '+' : '') + v.toFixed(4);
      section.append(el('p', {class: 'muted'}, fill(h.change > h.margin ? W.held_out_passed : W.held_out_failed,
        {change: n(h.change), margin: h.margin.toFixed(4)})));
    }
  } else section.append(el('div', {class: 'examples'}, r.examples.map(example)));
  return section;
}
// A Deep Dream's report: the outcome, what changed, how it went. The same
// report opens after the Dream (with Done) and from the Dream row.
function deepReport(c, current, done) {
  const r = c.result;
  return el('div', {class: 'report deep'},
    // The outcome is the heading (no "Dream finished" box).
    el('div', {class: 'report-head'},
      el('p', {class: 'report-title'}, outcomeText(r), el('span', {class: 'muted'}, ` · ${W.reason[r.reason]}`)),
      el('p', {class: 'muted'}, `${W.dream_kinds.deep.title} · ${when(c.ended_at)} · took ${duration(c.ended_at - c.started_at)}`),
      r.outcome === 'adopted' && r.in_use === false
        ? el('p', {class: 'muted'}, fill(W.undone, {version: r.version, current: current?.label || 'Original'})) : null),
    whatChanged(c),
    cycleView(c, W.how_it_went),
    done ? el('p', {class: 'report-done'}, el('button', {type: 'button', class: 'done', onclick: done}, W.done)) : null);
}
// A Quick Dream's report: the preferences it learned. View all expands the
// card in place with everything in the prompt now (no side panel).
const expanded = new Set();   // 'quick:<dream id>' and 'now:<model id>' shown open
function toggle(key) { expanded.has(key) ? expanded.delete(key) : expanded.add(key); render(); }
function preferenceList(lessons) {
  return lessons.length
    ? el('ul', {class: 'lesson-list'}, lessons.map(l => el('li', {}, el('span', {class: 'text'}, l.text),
      l.count > 1 ? el('span', {class: 'muted count'}, `×${l.count}`) : null)))
    : el('p', {class: 'empty'}, W.preferences_empty);
}
function quickReport(entry, {card = false, done = null, lessons = []} = {}) {
  const added = entry.changes?.added || [];
  const key = `quick:${entry.id}`, open = expanded.has(key);
  return el('div', {class: `report quick${card ? ' card' : ''}`},
    el('p', {class: 'report-title'}, W.quick_report_title, el('span', {class: 'muted'}, ` · ${when(entry.at)}`)),
    // What it read and found; why it failed; or what it added.
    entry.error ? el('p', {class: 'bad'}, W.quick_failed_card[entry.error] || W.quick_failed_card.unreadable)
      : added.length ? el('p', {}, plural(added.length, 'new_preferences'), el('span', {class: 'muted'}, ` · ${plural(entry.lessons, 'preferences')}`))
      : el('p', {}, W.no_new_preferences, el('span', {class: 'muted'}, ` · ${plural(entry.lessons, 'preferences')}`)),
    entry.read != null && !entry.error ? el('p', {class: 'muted'}, fill(W.quick_read_found, {read: plural(entry.read, 'conversations'), found: plural(entry.found || 0, 'moments')})) : null,
    added.length ? el('ul', {class: 'changes'}, added.map(t => el('li', {class: 'added'},
      el('span', {class: 'sign'}, '+'), el('span', {class: 'text'}, t,
        quickExample(entry, t))))) : null,
    open ? el('div', {class: 'all-preferences'}, el('h4', {class: 'section-title'}, W.preferences_title), preferenceList(lessons)) : null,
    el('p', {class: 'report-actions'},
      el('button', {type: 'button', class: 'text-action', 'aria-expanded': String(open), onclick: () => toggle(key)},
        open ? W.view_less : W.view_all),
      done ? el('button', {type: 'button', class: 'done', onclick: done}, W.done) : null));
}
// What a new preference changes: the same request, before and after it was
// in the prompt, on the conversation it came from. Honest when nothing changed.
function quickExample(entry, lesson) {
  const e = (entry.examples || []).find(x => x.lesson === lesson);
  if (!e) return null;
  const key = `${entry.id}:${lesson}`;
  return el('div', {class: 'quick-example'},
    el('p', {class: 'example-where muted'}, W.quick_example_where),
    el('p', {class: 'example-task'}, fill(W.quick_example_asked, {asked: short(e.asked, 140)}),
      e.feedback ? el('span', {class: 'muted'}, ` · ${fill(e.said_by === 'tool' ? W.quick_example_tool : W.quick_example_said, {said: short(e.feedback, 100)})}`) : null),
    e.changed
      ? el('div', {class: 'pair'}, answerBox('before', e.before, e.after, `${key}:before`),
          answerBox('after', e.after, e.before, `${key}:after`))
      : el('p', {class: 'example-same muted'}, W.quick_example_same));
}
// The Dream row: newest first; unseen ones say "new".
const selectedDream = {};     // model id → dream id shown under the row
function dreamRow(id, dreams, chosen) {
  return el('div', {class: 'dream-row', role: 'tablist', 'aria-label': W.dreams_title}, dreams.map(d => el('button', {
    type: 'button', role: 'tab', class: `chip ${d.kind}${d.id === chosen ? ' chosen' : ''}`, 'aria-selected': String(d.id === chosen),
    onclick: () => { selectedDream[id] = d.id; if (!d.seen) control('seen', {model: id, id: d.id}); else render(); }},
    el('span', {class: 'chip-kind'}, d.kind === 'quick' ? W.chip_quick : W.chip_deep),
    el('span', {class: 'chip-what'}, d.kind === 'quick'
      ? (d.error ? W.chip_failed : d.changes.added.length ? `+${d.changes.added.length}`
        : d.changes.reworded.length ? 'reworded' : `${W.chip_no_change}${d.repeat > 1 ? ` ×${d.repeat}` : ''}`)
      : d.outcome === 'adopted' ? d.version || 'adopted' : d.outcome === 'kept' ? 'not used' : 'stopped'),
    el('span', {class: 'chip-when'}, when(d.at)),
    d.seen ? null : el('span', {class: 'chip-new'}, W.chip_new))));
}
// The report of a Dream from the row: Deep reports load once in full.
function rowReport(id, ml, d) {
  const current = ml.versions?.find(v => v.current);
  if (d.kind === 'quick') return quickReport(d, {lessons: ml.lessons || []});
  const c = d.id === ml.last?.id ? ml.last : dreamRecords.get(d.id);
  if (!c) { loadDream(d.id); return el('p', {class: 'empty'}, W.loading_report); }
  // An older Dream's version: "Adopted v1", not "Now using v1", once v2 runs.
  const result = c.result && {...c.result, in_use: c.result.in_use ?? (current?.cycle === d.id)};
  return deepReport({...c, result}, current, null);
}
// Older Deep Dreams' records for the row, kept apart from the practice
// list's "Show all" (which drops its own copy on "Show fewer").
const dreamRecords = new Map(), dreamLoading = new Set();
async function loadDream(id) {
  if (dreamLoading.has(id)) return;
  dreamLoading.add(id);
  try {
    const r = await fetch(`/api/cycles/${encodeURIComponent(id)}`, {cache: 'no-store'});
    if (r.ok) dreamRecords.set(id, await r.json());
  } finally {
    dreamLoading.delete(id);
    render();
  }
}
function nowLine(id, ml) {
  const key = `now:${id}`, open = expanded.has(key);
  const waiting = ml?.waiting || {total: 0, lessons: []};
  const line = el('p', {class: 'now-line'},
    fill(W.now_line, {prefs: plural((ml?.lessons || []).length, 'preferences'), waiting: plural(waiting.total, 'skills')}),
    // A Quick Dream that found nothing new leaves no card: this line says it ran.
    ml?.quick_checked?.nothing_new ? el('span', {class: 'muted'}, ` · ${ml.quick_checked.read
      ? fill(W.quick_checked_read, {when: when(ml.quick_checked.at), read: plural(ml.quick_checked.read, 'conversations'), found: plural(ml.quick_checked.found || 0, 'moments')})
      : fill(W.quick_checked, {when: when(ml.quick_checked.at)})}`) : null,
    // A Quick Dream that could not read, or not start: said, not "nothing new".
    ml?.quick_checked?.error ? el('span', {class: 'bad'}, ` · ${fill(W.quick_failed[ml.quick_checked.error] || W.quick_failed.unreadable, {when: when(ml.quick_checked.at), count: ml.quick_checked.failed || ''})}`) : null,
    ' · ', el('button', {type: 'button', class: 'view-learned', 'aria-expanded': String(open), onclick: () => toggle(key)},
      open ? W.view_less : W.view));
  if (!open) return line;
  const waitingList = el('details', {class: 'waiting'},
    el('summary', {}, `${W.waiting_title} (${waiting.total})`),
    el('p', {class: 'muted'}, W.waiting_about),
    waiting.lessons.length ? preferenceList(waiting.lessons) : el('p', {class: 'empty'}, W.waiting_empty));
  waitingList.open = waitingOpen;
  waitingList.addEventListener('toggle', () => { waitingOpen = waitingList.open; });
  return el('div', {class: 'learned'}, line,
    el('section', {}, el('h4', {class: 'section-title'}, `${W.preferences_title} (${(ml?.lessons || []).length})`),
      el('p', {class: 'muted'}, W.preferences_about), preferenceList(ml?.lessons || [])),
    el('section', {}, waitingList));
}
let waitingOpen = false;      // the waiting list is expanded

// ---- the cycle ----
function lossChart(losses) {
  if (losses.length < 2) return null;
  const w = 240, h = 40, max = Math.max(...losses), min = Math.min(...losses);
  const points = losses.map((v, i) => `${(i / (losses.length - 1) * w).toFixed(1)},${(h - 2 - (v - min) / (max - min || 1) * (h - 4)).toFixed(1)}`).join(' ');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('width', w); svg.setAttribute('height', h);
  svg.setAttribute('class', 'loss'); svg.setAttribute('aria-label', 'Training loss by step');
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  line.setAttribute('points', points);
  svg.append(line);
  return svg;
}
function bar(p) {
  if (!p || !p.total) return null;
  return el('div', {class: 'bar'}, el('span', {style: `width:${Math.min(100, p.done / p.total * 100)}%`}));
}
function practiceRows(lines) {
  return el('div', {class: 'rows'}, lines.map(l => el('div', {class: 'row', 'data-kept': String(l.kept)},
    el('span', {class: l.kept ? 'good' : 'muted'}, l.kept ? '✓' : '–'),
    el('span', {class: 'text'}, short(l.request, 110)),
    el('span', {class: 'why'}, l.kept ? '' : W.practice_drop[l.drop] || l.drop))));
}
function stageDetail(c, s) {
  const p = s.progress;
  switch (s.key) {
    case 'read': return p ? `${p.done} of ${plural(p.total, 'conversations')} · ${c.read.signals.total} signals` : '';
    case 'practice': return c.practice.written ? `${c.practice.kept} of ${plural(c.practice.written, 'scenes')} kept` : '';
    case 'train': return s.state === 'done' ? `${plural(c.train.total, 'steps')} · ${c.train.samples.practice} practice + ${c.train.samples.rehearsal} rehearsal`
      // Step 0 is a real position (the first step is running), not "nothing".
      : s.state !== 'ahead' && c.train.total ? `step ${c.train.step || 0} of ${c.train.total}${s.state === 'now' && s.eta_seconds != null ? ` · about ${duration(s.eta_seconds)} left` : ''}` : '';
    case 'check': return p?.done ? `${p.done} of ${plural(p.total, p.unit || 'checks')}` : '';
    case 'result': return c.result ? `${outcomeText(c.result)} · ${W.reason[c.result.reason]}` : '';
  }
  return '';
}
function stageBody(c, s) {
  const body = el('div', {class: 'stage-body'}, bar(s.state === 'now' ? s.progress : null));
  if (s.key === 'read') {
    // One headline, then where the signals came from in plain words.
    const counts = c.read.signals.by_type, total = c.read.signals.total;
    const kinds = types => types.filter(t => counts[t]).map(t => {
      const [one, many] = W.signal_kind[t];
      return `${counts[t]} ${counts[t] === 1 ? one : many}`;
    }).join(', ');
    const groups = [['user', ['correction', 'retry', 'approval']], ['tool', ['tool_error', 'failed_result', 'outcome']]]
      .map(([source, types]) => [source, kinds(types)]).filter(([, text]) => text);
    body.append(...[
      total ? el('p', {class: 'read-headline'}, fill(W.read_headline, {
        signals: plural(total, 'signals'),
        conversations: plural(c.read.conversations.done, 'conversations'),
      })) : el('p', {}, 'No signals yet'),
      groups.length ? el('dl', {class: 'signal-groups'}, groups.map(([source, text]) => [
        el('dt', {}, W.signal_source[source]), el('dd', {}, text)])) : null,
      el('p', {class: 'muted'}, `${c.read.successes.confirmed} things it already did right are kept aside, to check it does not lose them.`),
    ].filter(Boolean));
  } else if (s.key === 'practice') {
    const full = fullRecords.get(c.id);
    const lines = full ? full.practice.lines : c.practice.recent;
    // The list's toggle sits above it, in the same place open or closed,
    // so the long list never needs a scroll down to collapse.
    if (full && lines.length > 5) {
      body.append(el('button', {type: 'button', class: 'show-all', onclick: () => { fullRecords.delete(c.id); render(); }}, 'Show fewer'));
    } else if (!full && c.practice.written > lines.length) {
      const state = loadingFull.get(c.id);
      body.append(state === 'loading'
        ? el('button', {type: 'button', class: 'show-all loading', disabled: true, 'aria-busy': 'true'},
            el('span', {class: 'spinner', 'aria-hidden': 'true'}), `Loading all ${c.practice.written}…`)
        : el('button', {type: 'button', class: 'show-all', onclick: () => loadFull(c.id)},
            state === 'failed' ? 'Could not load · try again' : `Show all ${c.practice.written}`));
    }
    body.append(lines.length ? practiceRows(lines) : el('p', {class: 'muted'}, 'No practice yet'));
  } else if (s.key === 'measure') {
    // Its own step: sizes tried, the limit, no loss chart or step counter.
    body.append(el('p', {class: 'muted'}, W.measure_about),
      s.sizes?.length ? el('p', {}, fill(W.measure_fits, {sizes: s.sizes.join(' → ')}), ' ✓',
        s.limit ? el('span', {class: 'muted'}, ` · ${fill(W.measure_limit, {limit: s.limit})}`) : null) : null,
      s.progress?.total ? el('p', {class: 'muted'}, fill(s.progress.longer ? W.measure_longer : W.getting_ready, s.progress)) : null);
  } else if (s.key === 'train') {
    const losses = c.train.losses;
    // lossChart is null below two steps; append() would print "null".
    const step = c.train.step || 0, total = c.train.total;
    // Before the first loss: where training stands, not "Not started".
    // Measuring what fits before the first step: getting ready, not "Not started".
    const measuring = s.progress?.unit === 'calibration' && !losses.length;
    const waiting = s.state === 'done' ? `${plural(total || 0, 'steps')} trained`
      : measuring ? fill(W.getting_ready, s.progress)
      : !total || s.state === 'ahead' ? 'Not started'
      : s.state === 'now' ? `Working on step ${step + 1} of ${total} · the loss line starts after it`
      : `At step ${step} of ${total} · carries on where it stopped`;
    body.append(...[lossChart(losses)].filter(Boolean), losses.length ? el('p', {class: 'muted'}, `loss ${losses[0].toFixed(3)} → ${losses.at(-1).toFixed(3)} over ${plural(losses.length, 'steps')}`) : el('p', {class: 'muted'}, waiting));
  } else if (s.key === 'check') {
    body.append(el('div', {class: 'rows checks'}, ...c.checks.map(k => el('div', {class: 'row'},
      el('span', {}, W.check[k.key]),
      el('span', {},
        k.missing ? el('span', {class: 'muted'}, 'not measured in this run')
          : k.state !== 'done' ? el('span', {class: 'muted'}, k.state === 'now' ? 'checking…' : '')
          : k.unit === 'judged' ? improvedText(k.judged)
          : k.unit === 'health' ? (k.passed === false ? (W.health_fail[k.text] || k.text) : k.passed ? W.health_ok : k.text === 'not_reached' ? el('span', {class: 'muted'}, W.health_not_reached) : '')
          : k.limit != null ? (k.after != null ? fill(W.check_change, {change: value(k.unit, k.after), limit: value(k.unit, k.limit)}) : el('span', {class: 'muted'}, W.check_not_measured))
          : `${value(k.unit, k.before)} → ${value(k.unit, k.after)}`,
        k.passed === true ? el('span', {class: 'good'}, ' ✓') : k.passed === false ? el('span', {class: 'bad'}, ' ✗')
          : k.state === 'done' ? el('span', {class: 'muted'}, ' –') : null))),
      // All of them must pass; said once, not as a tag on every row.
      c.checks.length ? el('p', {class: 'muted checks-rule'}, W.checks_rule) : null));
  } else if (s.key === 'result' && c.result) {
    if (c.result.improved?.count != null) body.append(el('p', {}, improvedText(c.result.improved)));
  }
  return body;
}
function cycleView(c, title) {
  const section = el('section', {class: 'cycle'}, el('h3', {class: 'section-title'}, title));
  const marks = {done: '✓', now: '●', waiting: '‖', skipped: '–', ahead: '·'};
  for (const s of c.stages) {
    const summary = [el('span', {class: 'mark'}, marks[s.state]), el('span', {class: 'stage-name'}, W.stage[s.key]), el('span', {class: 'detail'}, stageDetail(c, s))];
    const node = remembered(`${c.id}:${s.key}`, summary, stageBody(c, s), s.state === 'now');
    node.className = 'stage';
    node.dataset.state = s.state;
    section.append(node);
  }
  return section;
}
async function loadFull(id) {
  if (loadingFull.get(id) === 'loading') return;
  // The full record takes a second or two through a slow link: say so.
  loadingFull.set(id, 'loading');
  render();
  try {
    const r = await fetch(`/api/cycles/${encodeURIComponent(id)}`, {cache: 'no-store'});
    if (!r.ok) throw new Error(String(r.status));
    fullRecords.set(id, await r.json());
    loadingFull.delete(id);
  } catch {
    loadingFull.set(id, 'failed');
  }
  render();
}

// ---- a model ----
const learns = (ml, inv) => Boolean(ml?.current || ml?.last) || (inv && inv.learning_level !== 'off' && inv.support !== 'unsupported');
function quiet(ml, inv, text) {
  return el('section', {class: 'changed'}, el('div', {class: 'finished kept'},
    el('p', {class: 'finished-title'}, text[0]), text[1] ? el('p', {class: 'finished-outcome muted'}, text[1]) : null));
}
function modelView(id, ml, inv, paused = false) {
  const level = inv?.learning_level || 'new';
  const unsupported = inv?.support === 'unsupported';
  // What it learns from is Songe's decision (new conversations, older
  // practice replayed); the viewer only chooses whether Dreams start alone.
  const auto = inv?.auto_dream ?? level !== 'off';
  const autoSwitch = el('button', {type: 'button', class: 'auto', role: 'switch', 'aria-checked': String(auto),
    title: W.auto_dream_title, onclick: () => control('auto_dream', {model: id, on: !auto})},
    el('span', {class: 'track', 'aria-hidden': 'true'}, el('span', {class: 'knob'})), W.auto_dream);
  const current = ml?.versions?.find(v => v.current);
  // One button follows the Dream: start it, pause it, resume it. Pausing is
  // service-wide (every model's learning stops), as the rail's Paused line says.
  const mode = paused ? 'resume' : ml?.current ? 'pause' : 'dream';
  const dreamButton = el('button', {type: 'button', class: `dream ${mode}`, 'data-mode': mode,
    'aria-haspopup': mode === 'dream' ? 'menu' : null, 'aria-expanded': mode === 'dream' ? String(dreamMenu === id) : null},
    W.dream_button[mode]);
  // Dream asks which: Quick (lessons in the prompt) or Deep (training).
  dreamButton.onclick = mode === 'dream'
    ? () => { dreamMenu = dreamMenu === id ? null : id; render(); }
    : () => control(mode, {});
  dreamButton.disabled = mode === 'dream' && (level === 'off' || unsupported);
  const dream = el('span', {class: 'dream-wrap'}, dreamButton,
    mode === 'dream' && dreamMenu === id ? el('div', {class: 'dream-menu', role: 'menu'},
      ['quick', 'deep'].map(kind => el('button', {type: 'button', role: 'menuitem', class: `dream-kind ${kind}`,
        onclick: () => { dreamMenu = null; control('dream', {model: id, kind}); }},
        el('span', {class: 'kind-title'}, W.dream_kinds[kind].title, el('span', {class: 'kind-time'}, ` · ${W.dream_kinds[kind].time}`)),
        el('span', {class: 'kind-about'}, W.dream_kinds[kind].about)))) : null);
  const running = ml?.current ? 'deep' : ml?.quick_running ? 'quick' : null;
  if (!running && atDashboard.has(id)) { atDashboard.delete(id); rememberDashboard(); }
  const away = Boolean(running) && atDashboard.has(id);
  const node = el('section', {class: 'model'},
    el('div', {class: 'model-head'},
      el('h2', {class: 'model-name'}, inv?.name || id),
      el('span', {class: 'version'}, current ? `${current.label} in use` : ''),
      el('span', {class: 'spacer'}), unsupported ? null : autoSwitch, unsupported ? null : dream),
    noticeView(),
    // Measuring this machine (not a Dream), and how it ended.
    ml?.measuring_machine ? el('div', {class: 'notice', role: 'status', 'data-failure': 'false'},
      el('p', {}, fill(W.measuring_machine, {name: inv?.name || id}),
        ml.measuring_machine.total ? el('span', {class: 'muted'}, ` · ${fill(W.measuring_progress, ml.measuring_machine)}`) : null)) : null,
    !ml?.measuring_machine && ml?.machine_measured ? el('div', {class: 'notice', role: 'status', 'data-failure': String(ml.machine_measured.support === 'unsupported')},
      el('p', {}, fill(ml.machine_measured.support === 'unsupported' ? W.measured_cannot : W.measured_can,
        {name: inv?.name || id, reason: ml.machine_measured.reason || ''})),
      el('button', {type: 'button', class: 'notice-close', 'aria-label': 'Close',
        onclick: () => control('seen', {model: id, id: 'measured'})}, '×')) : null,
    // A pressed Deep Dream that stopped before training says so until closed.
    ml?.stopped && !ml?.current ? stoppedNote(id, ml.stopped) : null,
    // A pressed Deep Dream between its runs: it keeps showing, with its step.
    ml?.deep_waiting ? el('div', {class: 'notice', role: 'status', 'data-failure': 'false'}, el('p', {}, ml.deep_waiting)) : null,
    // A Dream running: its progress view (with a way back to all Dreams),
    // or, once the viewer went back, a slim live card on top of the dashboard.
    away ? runningCard(id, ml, running) : null,
    running === 'quick' && !away ? [backLink(id), quickRunning(ml.quick_running)] : null);
  const dreams = ml?.dreams || [];
  const latest = dreams[0];
  const seen = d => () => control('seen', {model: id, id: d.id});
  if (unsupported) node.append(quiet(ml, inv, [W.wait.unsupported, inv.reason && `It is ${inv.reason}.`]));
  else if (running === 'deep' && !away) {
    // A Deep Dream running: the learning view.
    node.append(backLink(id), el('section', {class: 'changed'}, dreaming(ml.current)), cycleView(ml.current, 'This Dream'));
  } else if (running === 'quick' && !away) {
    // A Quick Dream running: its view above is the page.
  } else if (latest && !latest.seen && latest.kind === 'deep' && ml.last?.id === latest.id) {
    // Just finished and not seen yet: its report, until Done.
    node.append(deepReport(ml.last, ml.versions?.find(v => v.current), seen(latest)));
  } else if (!dreams.length) {
    node.append(level === 'off'
      ? quiet(ml, inv, ['Not learning yet', 'Turn on auto-dream to let it learn from your new conversations.'])
      : quiet(ml, inv, ['No Dream yet', `${plural(ml?.pending?.conversations || 0, 'conversations')} collected so far. ${auto ? 'It dreams when your computer is idle, or press Dream.' : 'Auto-dream is off: press Dream to run one.'}`]));
    if (level !== 'off') node.append(nowLine(id, ml));
  } else {
    // The dashboard: a new Quick Dream on top, the Dream row, the chosen
    // Dream's report, what it knows now.
    const card = latest && !latest.seen && latest.kind === 'quick';
    if (card) {
      node.append(quickReport(latest, {card: true, done: seen(latest), lessons: ml.lessons || []}));
    }
    // While the new Quick Dream is the card on top, the row opens on the
    // Dream before it, so its report is not shown twice.
    const chosen = dreams.find(d => d.id === selectedDream[id]) || (card && dreams[1]) || dreams[0];
    node.append(
      el('section', {class: 'dreams'}, el('h3', {class: 'section-title'}, W.dreams_title), dreamRow(id, dreams, chosen.id)),
      el('section', {class: 'chosen-report'}, rowReport(id, ml, chosen)),
      nowLine(id, ml));
  }
  if (ml?.versions?.length > 1) node.append(el('p', {class: 'versions'}, 'Versions · ', ml.versions.map((v, i) => [i ? ' · ' : '',
    v.current ? el('b', {}, `${v.label} in use`)
      : el('button', {type: 'button', class: 'restore', onclick: () => control('undo', {model: id, version: v.id})}, fill(W.restore, {version: v.label}))])));
  return node;
}

// The worker's reason in the note's words ("Waiting for memory" → "waiting for memory").
function stoppedDetail(detail) {
  const known = Object.entries(W.dream_stopped_why).find(([k]) => String(detail || '').startsWith(k));
  if (known) return known[1];
  const d = String(detail || 'it ended early').replace(/\.$/, '');
  return d.charAt(0).toLowerCase() + d.slice(1);
}
function stoppedNote(id, stopped) {
  return el('div', {class: 'notice', role: 'status', 'data-failure': 'true'},
    el('p', {}, `${fill(W.dream_stopped[stopped.reason] || W.dream_stopped.reading, {detail: stoppedDetail(stopped.detail)})} ${stopped.kept ? fill(W.dream_stopped_kept, {kept: stopped.kept}) + ' ' : ''}`, el('span', {class: 'muted'}, when(stopped.at)), ' · ',
      el('button', {type: 'button', class: 'notice-action', title: stopped.detail || '',
        onclick: () => control('dream', {model: id, kind: 'deep'})}, W.dream_stopped_retry)),
    el('button', {type: 'button', class: 'notice-close', 'aria-label': 'Close',
      onclick: () => control('seen', {model: id, id: 'stopped'})}, '×'));
}
// While a Dream runs the viewer may go back to all Dreams; remembered per
// model for this browser session, forgotten when the Dream ends.
const atDashboard = new Set((() => { try { return JSON.parse(sessionStorage.getItem('songe:atDashboard') || '[]'); } catch { return []; } })());
function rememberDashboard() {
  try { sessionStorage.setItem('songe:atDashboard', JSON.stringify([...atDashboard])); } catch { /* no storage: this page only */ }
}
function backLink(id) {
  return el('button', {type: 'button', class: 'back-link text-action', onclick: () => { atDashboard.add(id); rememberDashboard(); render(); }}, W.back_to_dreams);
}
// The running Dream as one live line on top of the dashboard.
function runningCard(id, ml, kind) {
  let text;
  if (kind === 'deep') {
    const c = ml.current, i = c.stages.findIndex(s => s.key === c.stage), stage = c.stages[i], p = stage.progress;
    text = !c.wait && p?.unit === 'calibration' ? `${W.chip_deep} Dream · ${fill(W.getting_ready_title, p)}` : [W.chip_deep + ' Dream', c.wait ? W.wait[c.wait.code] : W.stage[c.stage],
      !c.wait && p && p.total ? `${p.done} of ${plural(p.total, p.unit)}` : `step ${i + 1} of ${c.stages.length}`,
      !c.wait && stage.eta_seconds != null ? `about ${duration(stage.eta_seconds)} left` : null].filter(Boolean).join(' · ');
  } else {
    const q = ml.quick_running, reading = q.stage === 'read';
    text = [W.quick_title[q.kind] || W.quick_title.quick, reading ? W.quick_steps.read : W.quick_steps.prompt,
      reading && q.total ? `${q.done} of ${plural(q.total, 'conversations_read')}` : null,
      reading && q.eta_seconds != null ? `about ${duration(q.eta_seconds)} left` : null].filter(Boolean).join(' · ');
  }
  return el('div', {class: 'running-card', role: 'status'},
    el('span', {class: 'tick'}, '●'), el('span', {class: 'running-text'}, text),
    el('button', {type: 'button', class: 'text-action', onclick: () => { atDashboard.delete(id); rememberDashboard(); render(); }}, W.view_progress));
}
// A Quick Dream running: one line on top of the dashboard.
// A Quick Dream (or idle reading) under way: the Deep Dream view, shallower.
// Three steps; reading has a bar and a time left (it stops at its budget).
function quickRunning(q) {
  const reading = q.stage === 'read';
  const steps = ['read', 'pick', 'prompt'];
  const at = reading ? 0 : 2;
  const share = reading ? (q.total ? q.done / q.total : 0) : 1;
  const left = q.eta_seconds != null && reading ? `about ${duration(q.eta_seconds)} left` : null;
  const limit = q.budget && reading ? fill(W.quick_budget, {limit: duration(q.budget)}) : null;
  return el('div', {class: 'finished running quick-running'},
    el('p', {class: 'finished-title'}, el('span', {class: 'tick'}, '●'), ` ${W.quick_title[q.kind] || W.quick_title.quick}`,
      el('span', {class: 'muted'}, ` · started ${when(q.started_at)}`)),
    el('p', {class: 'finished-outcome'}, [
      reading ? W.quick_steps.read : q.stage === 'compare' ? W.quick_steps.compare : W.quick_steps.prompt,
      reading && q.total ? `${q.done} of ${plural(q.total, 'conversations_read')}` : null,
      left, limit,
    ].filter(Boolean).join(' · ')),
    el('div', {class: 'steps', 'aria-hidden': 'true', style: 'grid-template-columns:repeat(3,1fr)'}, steps.map((key, i) =>
      el('span', {title: W.quick_steps[key]}, el('i', {style: `width:${i < at ? 100 : i === at ? Math.round(share * 100) : 0}%`})))),
    // A large backlog read in the background (a first run): say it is
    // catching up, so it is not taken for a slow Quick Dream.
    q.kind === 'read' && q.total >= 10 ? el('p', {class: 'quick-reading muted'}, W.catching_up) : null,
    // What it is doing: the conversation it reads, and what it found so far.
    q.reading ? el('p', {class: 'quick-reading'}, fill(W.quick_reading, {request: q.reading})) : null,
    el('p', {class: 'quick-found-title'}, q.found_count ? fill(W.quick_found, {signals: plural(q.found_count, 'signals')}) : W.quick_found_none),
    q.found?.length ? el('ul', {class: 'quick-found'}, q.found.slice().reverse().map(f => el('li', {},
      el('span', {class: 'found-kind'}, W.found_kind[f.type] || f.type),
      el('span', {class: 'found-text'}, f.text)))) : null);
}

// ---- the model list in the rail ----
let selected = null;          // model id the viewer picked
let showAll = false;          // the "more" list is open
let navOpen = false;          // phone: the list is open
function railState(ml, inv) {
  if (inv?.support === 'unsupported') return ["Can't learn here", 'muted'];
  if (ml?.current) {
    const c = ml.current;
    return c.wait ? [`Waiting · ${W.wait[c.wait.code]}`, 'warn'] : [`Dreaming · ${W.stage[c.stage]}`, 'live'];
  }
  if (ml?.last?.result) {
    const r = ml.last.result;
    return r.outcome === 'adopted' ? [r.in_use === false ? outcomeText(r) : `${r.version} · better on ${r.improved.count}`, ''] : ['Kept its version', ''];
  }
  if (inv?.learning_level === 'off') return ['Not learning', 'muted'];
  return [`Collecting · ${plural(ml?.pending?.conversations || 0, 'conversations')}`, ''];
}
function pickDefault(ids, models) {
  const dreaming = ids.find(id => models[id]?.current);
  if (dreaming) return dreaming;
  const done = ids.filter(id => models[id]?.last).sort((a, b) => (models[b].last.ended_at || 0) - (models[a].last.ended_at || 0));
  return done[0] || ids[0];
}
function railNav(ids, models, inventory, active, rest) {
  const item = id => {
    const [state, tone] = railState(models[id], inventory[id]);
    // Names only; a dot while it dreams (its state is on its page).
    const b = el('button', {type: 'button', class: 'nav-model', 'aria-current': id === active ? 'true' : null, 'data-tone': tone, title: state},
      el('span', {class: 'nav-name'}, inventory[id]?.name || id, tone === 'live' ? el('span', {class: 'nav-dot', 'aria-label': state}, ' ●') : null));
    b.onclick = () => { selected = id; navOpen = false; try { history.replaceState(null, '', `#${encodeURIComponent(id)}`); } catch (e) {} render(); };
    return b;
  };
  const toggle = el('button', {type: 'button', class: 'nav-toggle', 'aria-expanded': String(navOpen)},
    el('span', {class: 'nav-name'}, inventory[active]?.name || active));
  toggle.onclick = () => { navOpen = !navOpen; render(); };
  const more = rest.length ? el('button', {type: 'button', class: 'nav-more', 'aria-expanded': String(showAll)},
    showAll ? 'Fewer models' : `+ ${rest.length} more · learning off`) : null;
  if (more) more.onclick = () => { showAll = !showAll; render(); };
  return el('nav', {class: 'models-nav', 'aria-label': 'Models', 'data-open': String(navOpen)},
    el('p', {class: 'label'}, 'Models'),
    toggle,
    el('div', {class: 'nav-list'}, ids.map(item), showAll ? rest.map(item) : null, more));
}

let last = null;
function pausedView(models) {
  const out = {};
  for (const [id, ml] of Object.entries(models || {})) {
    const c = ml?.current;
    out[id] = !c ? ml : {...ml, current: {...c,
      wait: c.wait || {code: 'paused', have: null, need: null, unit: null},
      stages: c.stages.map(s => s.state === 'now' ? {...s, state: 'waiting'} : s)}};
  }
  return out;
}
function render() {
  if (!last) return;
  const {inventory, status} = last;
  // Paused is the service's own state and is known at once; the Dream's
  // record catches up a moment later. Show a running Dream as waiting on it.
  const models = status?.paused ? pausedView(last.models) : last.models;
  const all = [...new Set([...Object.keys(inventory), ...Object.keys(models)])];
  const shown = all.filter(id => learns(models[id], inventory[id]));
  const rest = all.filter(id => !shown.includes(id))
    .sort((a, b) => (inventory[a]?.support === 'unsupported') - (inventory[b]?.support === 'unsupported'));
  if (!all.includes(selected)) {
    const fromHash = decodeURIComponent(location.hash.slice(1));
    selected = all.includes(fromHash) ? fromHash : null;
  }
  const active = selected || pickDefault(shown.length ? shown : all, models);
  if (active && rest.includes(active)) showAll = true;
  const [now, next] = statusLines(models, inventory, status?.paused);
  $('now').textContent = now;
  $('next').textContent = next;
  $('paused').textContent = status?.paused ? 'Yes' : 'No';
  // One line while a Dream trains or measures: how busy the GPU is.
  const gpu = status?.gpu;
  $('gpu-line').hidden = !gpu;
  if (gpu) $('gpu').textContent = fill(gpu.memory != null ? W.gpu_line : W.gpu_busy,
    {busy: gpu.busy, memory: `${(gpu.memory / 2 ** 30).toFixed(1)} GB`});
  $('nav').replaceChildren(all.length ? railNav(shown, models, inventory, active, rest) : '');
  $('models').replaceChildren(active ? modelView(active, models[active], inventory[active], Boolean(status?.paused)) : el('p', {class: 'empty'}, 'No models yet'));
}
async function refresh() {
  let state;
  try {
    if (!W) W = await (await fetch('/assets/wording.json', {cache: 'no-store'})).json();
    state = await (await fetch('/api/state?view=cycle&revisions=%7B%7D', {cache: 'no-store'})).json();
  } catch (e) {
    $('now').textContent = 'Songe is not answering';
    $('next').textContent = 'Run songe serve in your terminal';
    return;
  }
  try {
    const sections = state.sections || {};
    last = {
      // A model whose record could not be built is null ("never hides the
      // rest"): it is left out here (live: LM Studio's embedding model
      // blanked the whole page).
      models: Object.fromEntries(Object.entries(sections.cycle?.models || {}).filter(([, m]) => m)),
      inventory: Object.fromEntries((sections.models?.models || []).map(m => [m.id, m])),
      status: sections.status,
    };
    render();
  } catch (e) {
    // Songe answered; the page itself failed. Say that, not "not answering".
    console.error(e);
    $('now').textContent = W?.page_error || 'The page hit an error · reload it';
    $('next').textContent = '';
  }
}
window.refresh = refresh;

// The site's paper: a one-time grayscale noise tile.
(() => {
  const paper = document.querySelector('.paper-grain'), tile = document.createElement('canvas'), side = 200;
  tile.width = tile.height = side;
  const ctx = tile.getContext('2d');
  if (!paper || !ctx) return;
  const grain = ctx.createImageData(side, side);
  let state = 846731;
  for (let i = 0; i < grain.data.length; i += 4) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    grain.data[i] = grain.data[i + 1] = grain.data[i + 2] = state & 255;
    grain.data[i + 3] = 255;
  }
  ctx.putImageData(grain, 0, 0);
  paper.style.backgroundImage = `url(${tile.toDataURL()})`;
})();
// Ask again 1.5 s after each answer, never while one is still coming: a
// slow answer must not pile requests up behind it.
async function poll() {
  await refresh();
  setTimeout(poll, 1500);
}
poll();
