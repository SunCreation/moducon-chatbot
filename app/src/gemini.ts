// 모두콘 안내 챗봇 — Gemini API 폴백 클라이언트 (사용자 API 키, 브라우저 직접 호출)
import { judgeAll, type JudgeResult } from "./judge";

export type { JudgeResult };
export interface GeminiMsg { role: "user" | "model"; text: string }

/** SSE 스트리밍 generateContent. 키는 사용자가 UI에서 입력해 로컬스토리지에 저장. */
export async function geminiStream(
  msgs: GeminiMsg[],
  apiKey: string,
  onToken: (t: string) => void,
  signal?: AbortSignal,
  model = "gemini-3.5-flash",
): Promise<string> {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: msgs.map((m) => ({
        role: m.role === "user" ? "user" : "model",
        parts: [{ text: m.text }],
      })),
    }),
    signal,
  });
  if (!res.ok) throw Object.assign(new Error(`gemini ${res.status}`), { status: res.status });
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let full = "";
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;
      try {
        const j = JSON.parse(payload);
        const parts = j.candidates?.[0]?.content?.parts ?? [];
        for (const p of parts) {
          if (typeof p.text === "string" && p.text && !p.thoughtSignature) {
            full += p.text;
            onToken(p.text);
          }
        }
      } catch { /* 불완전 라인 */ }
    }
  }
  return full;
}

/** LLM-as-a-Judge(Gemini): 한 턴(질문·근거·답변)을 루브릭별로 병렬 채점해 평균.
 *  기준·병렬·집계는 judge.ts 공통 로직 사용. */
export async function judgeTurn(
  question: string,
  sources: string,
  answer: string,
  apiKey: string,
  model = "gemini-3.5-flash",
): Promise<JudgeResult> {
  const call = async (prompt: string): Promise<string> => {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    });
    if (!res.ok) throw new Error(`judge ${res.status}`);
    const j = await res.json();
    return j.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  };
  return judgeAll(question, sources, answer, call);
}
