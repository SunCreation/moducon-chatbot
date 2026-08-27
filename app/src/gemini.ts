// 모두콘 안내 챗봇 — Gemini API 폴백 클라이언트 (사용자 API 키, 브라우저 직접 호출)
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

/** LLM-as-a-Judge: 한 턴(질문·근거·답변)을 근거 준수·환각·출처 관점에서 평가 */
export interface JudgeResult {
  grounded: boolean;      // 답변이 근거 조각에 기반하는가
  noHalluc: boolean;      // 자료에 없는 내용을 지어내지 않았는가
  cited: boolean;         // [ID] 근거 표시를 했는가
  refusal: boolean;       // 자료에 없을 때 없다고 답했는가 (해당 시)
  score: number;          // 0-100
  comment: string;        // 한두 문장 평어
}

export async function judgeTurn(
  question: string,
  sources: string,
  answer: string,
  apiKey: string,
  model = "gemini-3.5-flash",
): Promise<JudgeResult> {
  const prompt = [
    "당신은 RAG 챗봇 답변의 평가자입니다. 아래 [질문], [근거자료], [답변]을 읽고 다음 기준으로 JSON만 출력합니다.",
    "grounded: 답변 내용이 근거자료에서 나왔는가 (true/false)",
    "noHalluc: 근거에 없는 사실을 지어내지 않았는가 (true/false)",
    "cited: 답변 안에 근거 조각의 [ID] 표시가 있는가 (true/false)",
    "refusal: 근거에 답이 없어서 '없다'고 답한 경우 true, 그 외 false",
    "score: 0-100 정수 (grounded·noHalluc·cited 반영)",
    "comment: 한두 문장 평어 (한국어)",
    '출력 형식: {"grounded":bool,"noHalluc":bool,"cited":bool,"refusal":bool,"score":int,"comment":"..."} — JSON 외 텍스트 금지.',
    "",
    `[질문] ${question}`,
    "",
    `[근거자료] ${sources}`,
    "",
    `[답변] ${answer}`,
  ].join("\n");

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
  });
  if (!res.ok) throw new Error(`judge ${res.status}`);
  const j = await res.json();
  const text: string = j.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("평가 JSON 파싱 실패");
  const parsed = JSON.parse(m[0]) as JudgeResult;
  // 모델이 5점 만점으로 오해하는 경우 방어 — 0~5 범위면 100점 척도로 환산
  if (typeof parsed.score === "number" && parsed.score <= 5) {
    parsed.score = Math.round((parsed.score / 5) * 100);
  }
  return parsed;
}
