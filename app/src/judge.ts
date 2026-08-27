// 모두콘 안내 챗봇 — LLM-as-a-Judge 공통 로직
// 평가 기준·프롬프트·출력 파싱을 판정 엔진(ollama/gemini)이 공유한다.

export interface JudgeResult {
  grounded: boolean;      // 답변이 근거 조각에 기반하는가
  noHalluc: boolean;      // 자료에 없는 내용을 지어내지 않았는가
  cited: boolean;         // [ID] 근거 표시를 했는가
  refusal: boolean;       // 자료에 없을 때 없다고 답했는가 (해당 시)
  score: number;          // 0-100
  comment: string;        // 한두 문장 평어
}

export function buildJudgePrompt(question: string, sources: string, answer: string): string {
  return [
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
}

export function parseJudge(text: string): JudgeResult {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("평가 JSON 파싱 실패");
  const j = JSON.parse(m[0]) as Partial<JudgeResult>;
  // 모델이 5점 만점으로 오해하는 경우 방어 — 0~5 범위면 100점 척도로 환산
  let score = typeof j.score === "number" ? j.score : 0;
  if (score <= 5) score = Math.round((score / 5) * 100);
  return {
    grounded: j.grounded === true,
    noHalluc: j.noHalluc === true,
    cited: j.cited === true,
    refusal: j.refusal === true,
    score,
    comment: typeof j.comment === "string" ? j.comment : "",
  };
}
