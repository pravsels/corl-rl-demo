"""Live success-rate graph of the latest rl-agent run, in the browser.

Serves a page on localhost that re-reads the run folder's episodes.jsonl every 2 s
and shows only the newest run: when `rl-agent run` starts again, the graph starts
over. No extra packages.

    .venv/bin/python scripts/success_graph.py outputs/rl/dsrl-clean
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Success Rate</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e0;
--series:#2a78d6;--good:#0ca30c;--critical:#d03b3b;--warning:#fab219;--neutral:#a3a29b}
@media (prefers-color-scheme:dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e86;--grid:#2c2c2a;--series:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.4 -apple-system,BlinkMacSystemFont,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 16px;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:10px 14px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .s{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 8px;color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:12px;margin-bottom:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
#chart svg{display:block;width:100%;height:auto;overflow:visible}.legend svg{flex:none}
.axis text{fill:var(--muted);font-size:11px}.gridline{stroke:var(--grid);stroke-width:1}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--grid);
border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.15);display:none;max-width:260px}
#tip b{font-variant-numeric:tabular-nums}
.bar{display:flex;height:22px;border-radius:6px;overflow:hidden;gap:2px;margin-top:4px}
.bar div{min-width:2px}
</style></head><body><main>
<h1>Success rate, latest run</h1><p class="sub" id="sub"></p>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Success rate (moving average over the last <span id="win"></span> episodes)</h2>
<div class="legend">
<span><svg width="22" height="10"><line x1="0" y1="5" x2="22" y2="5" stroke="var(--series)" stroke-width="2"/></svg>Moving average</span>
<span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--good)"/></svg>Success</span>
<span><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="var(--critical)"/></svg>Safety stop</span>
<span><svg width="10" height="10"><path d="M5 1 L9 9 L1 9Z" fill="var(--warning)"/></svg>Timeout or wrong button</span>
<span><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="none" stroke="var(--ink2)" stroke-width="1.5"/></svg>Eval episode</span>
</div><div id="chart"></div></div>
<div class="card"><h2>Outcomes</h2><div id="mix"></div></div>
</main><div id="tip"></div>
<script>
const WIN=10, tip=document.getElementById('tip');
document.getElementById('win').textContent=WIN;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const kind=o=>o==='success'?'good':o==='safety'?'critical':'warning';
const label={success:'Success',safety:'Safety stop',timeout:'Timeout',wrong_button:'Wrong button'};
const pct=x=>x==null?'–':Math.round(x*100)+'%';
function draw(rows){
  const n=rows.length;
  document.getElementById('sub').textContent=`${RUN} · ${n} episode${n===1?'':'s'} · refreshes every 2 s`;
  const train=rows.filter(r=>!r.eval), ok=r=>r.outcome==='success';
  const lastW=train.slice(-WIN);
  const tiles=[
    ['Last '+WIN, pct(lastW.length?lastW.filter(ok).length/lastW.length:null), `${lastW.filter(ok).length}/${lastW.length} succeeded`],
    ['This run', pct(n?rows.filter(ok).length/n:null), `${rows.filter(ok).length}/${n} succeeded`],
    ['Started', n?new Date(rows[0].time).toLocaleTimeString('en-GB'):'–', n?`last episode ${new Date(rows[n-1].time).toLocaleTimeString('en-GB')}`:''],
    ['Safety stops', rows.filter(r=>r.outcome==='safety').length, `of ${n} episode${n===1?'':'s'}`],
  ];
  document.getElementById('tiles').innerHTML=tiles.map(t=>`<div class="tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join('');
  const W=1040,H=300,L=40,R=12,T=12,B=30,pw=W-L-R,ph=H-T-B;
  const x=i=>L+(n<=1?pw/2:i/(n-1)*pw), y=v=>T+(1-v)*ph;
  let g='';
  for(const v of [0,.25,.5,.75,1])g+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end">${v*100}%</text>`;
  const step=Math.max(1,Math.ceil(n/12));
  for(let i=0;i<n;i+=step)g+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">${i+1}</text>`;
  const rate=rows.map((r,i)=>{const w=rows.slice(0,i+1).filter(q=>!q.eval).slice(-WIN);return w.length?w.filter(ok).length/w.length:0});
  if(n>1)g+=`<polyline fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round" points="${rate.map((v,i)=>x(i)+','+y(v)).join(' ')}"/>`;
  rows.forEach((r,i)=>{const cx=x(i),cy=r.outcome==='success'?y(1):y(0),c=`var(--${kind(r.outcome)})`,s=`stroke="var(--surface)" stroke-width="2"`;
    g+= r.outcome==='success'?`<circle cx="${cx}" cy="${cy}" r="4.5" fill="${c}" ${s}/>`:
        r.outcome==='safety'?`<rect x="${cx-4.5}" y="${cy-4.5}" width="9" height="9" fill="${c}" ${s}/>`:
        `<path d="M${cx} ${cy-5} L${cx+5} ${cy+4} L${cx-5} ${cy+4}Z" fill="${c}" ${s}/>`;
    if(r.eval)g+=`<circle cx="${cx}" cy="${cy}" r="8" fill="none" stroke="var(--ink2)" stroke-width="1.5"/>`;
  });
  g+=`<line id="cross" y1="${T}" y2="${T+ph}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/><rect id="hit" x="${L}" y="${T}" width="${pw}" height="${ph}" fill="transparent"/>`;
  document.getElementById('chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" class="axis" role="img" aria-label="Success rate by episode">${g}</svg>`;
  const svg=document.querySelector('#chart svg'),hit=svg.querySelector('#hit'),cross=svg.querySelector('#cross');
  hit.onmousemove=e=>{if(!n)return;const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());
    const i=Math.max(0,Math.min(n-1,Math.round((q.x-L)/pw*(n-1))));const r=rows[i];
    cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('visibility','visible');
    tip.innerHTML=`<b>#${i+1}</b> ${r.eval?'(eval) ':''}· ${new Date(r.time).toLocaleTimeString('en-GB')}<br>Outcome: <b>${label[r.outcome]||r.outcome}</b>${r.time_to_press_s?` · ${r.time_to_press_s} s`:''}<br>Moving average: <b>${pct(rate[i])}</b>${r.reason?`<br>${r.reason}`:''}<br>Moved up to ${r.moved_max_deg}° · ${r.learner_updates} updates`;
    tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-280)+'px';tip.style.top=(e.clientY+14)+'px'};
  hit.onmouseleave=()=>{tip.style.display='none';cross.setAttribute('visibility','hidden')};
  const counts={};rows.forEach(r=>counts[r.outcome]=(counts[r.outcome]||0)+1);
  const order=['success','timeout','wrong_button','safety'].filter(k=>counts[k]);
  document.getElementById('mix').innerHTML=n?`<div class="legend">${order.map(k=>`<span>${label[k]} ${counts[k]} (${pct(counts[k]/n)})</span>`).join('')}</div><div class="bar">${order.map(k=>`<div style="flex:${counts[k]};background:var(--${kind(k)})" title="${label[k]} ${counts[k]}"></div>`).join('')}</div>`:'No episodes yet';
}
async function tick(){try{const r=await fetch('episodes',{cache:'no-store'});draw(await r.json())}catch(e){}}
tick();setInterval(tick,2000);
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default="outputs/rl/dsrl-clean")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    log = Path(args.run_dir) / "episodes.jsonl"
    page = PAGE.replace("${RUN}", args.run_dir.replace("`", ""))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/episodes"):
                rows = []
                if log.exists():
                    for line in log.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row.pop("trace", None)
                        if rows and (row.get("run", "") != rows[-1].get("run", "") if "run" in row
                                     else row.get("episode", 0) <= rows[-1].get("episode", 0)):
                            rows = []  # a new rl-agent run
                        rows.append(row)
                body, kind = json.dumps(rows).encode(), "application/json"
            else:
                body, kind = page.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{url}  ({log}; Ctrl+C stops)", flush=True)
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
