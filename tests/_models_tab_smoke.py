"""Models tab smoke test: live Hugging Face search + assessment in the window, decision tree, installed vaults with tuning info,
tuned settings reaching Model defaults and the engine command line. No downloads beyond config.json files."""
import sys, time, types, warnings, logging
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]; warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import harness_app as AP, harness_core as core, harness_paths as hp, chat
app = AP.App(); ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
inst = {app.hub_inst.item(i, "text"): app.hub_inst.item(i, "values") for i in app.hub_inst.get_children()}
chk("installed vaults listed with tuning status: " + str(inst.get("vault_gptoss_20b_mxf2")), "4.006 tok/s" in str(inst.get("vault_gptoss_20b_mxf2", "")))
app.hub_q.set("gpt-oss"); app._hub_search()
t0 = time.time()
while time.time() - t0 < 150:                     # let the background search + assessments run, pumping the UI queue
    app.update(); app._poll() if False else None; time.sleep(0.2)
    rows = app.hub_res.get_children()
    if rows and all(app.hub_res.item(r, "values")[3] != "checking ..." for r in rows[:6]): break
rows = app.hub_res.get_children(); print(f"   {len(rows)} results in {time.time() - t0:.0f}s:")
for r in rows[:6]: print("     ", r, app.hub_res.item(r, "values"))
chk("search returned models", len(rows) > 0 and "openai/gpt-oss-20b" in rows)
app.hub_res.selection_set("openai/gpt-oss-20b"); app._hub_select(); app.update()
kids = app.hub_tree.get_children(app.hub_tree.get_children()[0]) if app.hub_tree.get_children() else []
labels = [app.hub_tree.item(k, "text") for k in kids]; print("   tree:", labels)
chk("decision tree shown for gpt-oss-20b", any("Architecture" in l for l in labels) and any("Speed" in l for l in labels))
rec, notes = core.recommended_settings(hp.resolve_vault("vault_gptoss_20b_mxf2"))
chk("Model defaults include the tuned settings", rec.get("moe_prefetch") == 0 and rec.get("moe_vram") is True and any("Tuned on this machine" in n for n in notes))
a = types.SimpleNamespace(mode="auto", gpu_layers=0, ctx=8192, draft="none", temp=0, top_p=1, top_k=1, k=8, kv16=True, moe_prefetch=None, moe_vram=None)
args, how, _ = chat.pick_engine(hp.resolve_vault("vault_gptoss_20b_mxf2"), a)
chk("engine launched with the tuned settings (no --prefetch, VRAM on): " + how, "--prefetch" not in args and "--vram-mb" in args and "--kv16" in args)
app.after(200, app.destroy); app.mainloop(); print("MODELS TAB", "PASSED" if ok else "FAILED")
