import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import harness_core as c
cases = [
    ('<|channel|>analysis<|message|>Need 17*23.<|end|><|start|>assistant<|channel|>commentary to=functions.calculator <|constrain|>json<|message|>{"expression": "17*23"}',
     dict(thinking="Need 17*23.", answer="", calls=[("calculator", {"expression": "17*23"})], harmony=True)),
    ('<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>Answer is 391.',
     dict(thinking="think", answer="Answer is 391.", calls=[], harmony=True)),
    ('<|start|>assistant to=functions.clock<|channel|>commentary json<|message|>{}', dict(thinking="", answer="", calls=[("clock", {})], harmony=True)),
    ('<think>\nhmm\n</think>\n\nHi <tool_call>\n{"name": "clock", "arguments": {}}\n</tool_call>',
     dict(thinking="hmm", answer='Hi <tool_call>\n{"name": "clock", "arguments": {}}\n</tool_call>', calls=[("clock", {})], harmony=False)),
    ('<|channel|>analysis<|message|>part', dict(thinking="part", answer="", calls=[], harmony=True)),
    ('<think>\nstill thinking', dict(thinking="still thinking", answer="", calls=[], harmony=False)),
    ('plain answer', dict(thinking="", answer="plain answer", calls=[], harmony=False)),
]
ok = True
for raw, want in cases:
    got = c.parse_reply(raw); good = got == want; ok &= good
    print("PASS" if good else "FAIL", repr(raw[:50]), "" if good else f"\n   got  {got}\n   want {want}")
# streaming: after every token (special tokens arrive whole, text in small pieces) thinking/answer may only grow
import re
for raw in (cases[1][0], cases[3][0], cases[0][0]):
  toks = [t for t in re.split(r"(<\|[a-z_]+\|>|<think>|</think>|\s+)", raw) if t]; prev = {"thinking": "", "answer": ""}
  for i in range(1, len(toks) + 1):
    r = c.parse_reply("".join(toks[:i]))
    for k in prev:
        if not r[k].startswith(prev[k]): ok = False; print("FAIL stream prefix", k, repr(prev[k]), "->", repr(r[k]))
        prev[k] = r[k] if r[k].startswith(prev[k]) else prev[k]
print("stream prefixes monotone" if ok else "")
for v in ("vaults/vault_gptoss_120b_mxf2", "vaults/vault_06b_huff", "D:/models/vaults_parked/vault_hermes14b_huff", "D:/models/vaults_parked/vault_qwen3_30b_a3b_huff", "vaults/vault_llama1b_rans"):
    s, n = c.recommended_settings(v); print(v.split("/")[-1], s); print("   ", " | ".join(n[1:]))
print("PARSER TESTS", "PASSED" if ok else "FAILED")
