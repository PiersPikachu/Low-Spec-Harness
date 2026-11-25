"""window smoke test without a model: reasoning stream -> preview -> expand/collapse; history rendering; Model defaults button"""
import sys
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_app as A
app = A.App(); ok = True
def txt(): return app.tx.get("1.0", "end")
app._on_start("model"); app._on_thinking("Let me think about "); app._on_thinking("the question carefully. "); live = txt()
ok &= "Let me think about the question carefully." in live
app._on_tool("-> calculator({\"expression\": \"2+2\"})\n"); app._on_assistant("The answer is 4.")
t = txt(); P = "Let me think about the question carefully."; ok &= "Thought for 7 words" in t and t.count(P) == 1 and not app.tx.tag_ranges("think_live") and "The answer is 4." in t
print("stream -> preview:", "PASS" if ok else "FAIL"); print(t[-300:])
n = app._think_n; app._toggle_think(n); ok2 = txt().count(P) == 2; app._toggle_think(n); ok2 &= txt().count(P) == 1
print("expand / collapse:", "PASS" if ok2 else "FAIL")
app.conv = {"id": "x", "title": "t", "vault": "vault_gptoss_20b_mxf2", "messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "", "thinking": "use the tool", "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": {"expression": "1+1"}}}]},
    {"role": "tool", "content": "2", "meta": {"tool": "calculator"}}, {"role": "assistant", "content": "It is 2.", "thinking": "done"}]}
app._render(); r = txt(); ok3 = "Thought for 3 words" in r and "-> calculator" in r and "It is 2." in r
print("history render:", "PASS" if ok3 else "FAIL")
app.vars["vault"].set("vault_gptoss_120b_mxf2"); app._model_defaults()
ok4 = app.vars["temp"].get() == "1.0" and app.vars["top_p"].get() == "1.0" and app.vars["draft"].get() == "none" and app.vars["max_new"].get() == "4096" and app.vars["kv16"].get() is True
print("Model defaults (gpt-oss-120b):", "PASS" if ok4 else "FAIL", {k: app.vars[k].get() for k in ("temp", "top_p", "top_k", "max_new", "ctx", "think", "greedy", "draft")})
app.vars["vault"].set("vault_06b_huff"); app._model_defaults()
print("Model defaults (Qwen3-0.6B):", {k: app.vars[k].get() for k in ("temp", "top_p", "top_k", "max_new", "ctx", "think", "greedy", "draft")})
app._fill_config()      # discard, as Revert would
app.after(200, app.destroy); app.mainloop()
print("APP SMOKE", "PASSED" if ok and ok2 and ok3 and ok4 else "FAILED")
