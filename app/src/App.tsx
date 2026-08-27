import { useState, useRef, useEffect } from "react";
import { retrieve, buildPrompt, loadCorpus, type Retrieved } from "./rag";
import { chatStream, pingOllama, type ChatMsg } from "./ollama";
import { geminiStream, judgeTurn, type JudgeResult } from "./gemini";
import "./App.css";

interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: Retrieved[];
  question?: string;
  judge?: JudgeResult;
  feedback?: "up" | "down";
}

type Phase = "idle" | "embed" | "search" | "stream" | "error-ollama";

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([
    {
      role: "assistant",
      content:
        "안녕하세요. 모두콘 안내 챗봇입니다. 모두콘에 대해 무엇이든 물어보세요. 답은 공개 문서에서 뽑은 근거로만 드립니다.",
    },
  ]);
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);
  const [engine, setEngine] = useState<"local" | "gemini">("local");
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("gemini_key") ?? "");
  const [showSource, setShowSource] = useState<Retrieved[] | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    pingOllama().then(setOllamaOk);
    loadCorpus().catch(() => undefined); // 프리로드
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, phase]);

  async function ask() {
    const q = input.trim();
    if (!q || phase !== "idle") return;
    setInput("");
    setTurns((t) => [...t, { role: "user", content: q }]);

    try {
      setPhase("search");
      const hits = await retrieve(q, 3);
      const prompt = buildPrompt(q, hits);
      const messages: ChatMsg[] = [
        {
          role: "system",
          content:
            "당신은 모두의연구소 컨퍼런스 '모두콘'의 안내 도우미입니다. 주어진 자료에 근거한 내용만 답하고, 자료에 없는 정보는 '제가 가진 자료에는 없습니다'라고 답합니다. 근거 조각의 [ID]를 답에 표시합니다.",
        },
        { role: "user", content: prompt },
      ];

      const lastQ = q;
      setTurns((t) => [...t, { role: "assistant", content: "", sources: hits, question: lastQ }]);
      setPhase("stream");
      abortRef.current = new AbortController();
      let acc = "";
      const onPiece = (piece: string) => {
        acc += piece;
        setTurns((t) => {
          const copy = [...t];
          copy[copy.length - 1] = { role: "assistant", content: acc, sources: hits };
          return copy;
        });
      };
      if (engine === "gemini") {
        await geminiStream(
          [
            { role: "user", text: messages[0].content },
            { role: "user", text: messages[1].content },
          ],
          apiKey,
          onPiece,
          abortRef.current.signal,
        );
      } else {
        await chatStream(messages, onPiece, "qwen3.5:2b", abortRef.current.signal);
      }
      setPhase("idle");
      // LLM-as-a-Judge (gemini 키가 있을 때 자동 평가)
      if (apiKey) {
        try {
          const src = hits.map((h) => `[${h.chunk.id}] ${h.chunk.text}`).join("\n");
          const verdict = await judgeTurn(lastQ, src, acc, apiKey);
          setTurns((t) => {
            const copy = [...t];
            const li = copy.length - 1;
            copy[li] = { ...copy[li], judge: verdict };
            return copy;
          });
        } catch { /* 평가 실패는 답변을 해치지 않음 */ }
      }
    } catch (e: unknown) {
      const status = (e as { status?: number }).status;
      if (status === 403) {
        setPhase("error-ollama");
        setOllamaOk(false);
      } else {
        setPhase("error-ollama");
        setOllamaOk(false);
      }
      setTurns((t) => t.filter((x) => x.content !== ""));
    }
  }

  function stop() {
    abortRef.current?.abort();
    setPhase("idle");
  }

  function setFeedback(i: number, v: "up" | "down") {
    setTurns((t) => {
      const copy = [...t];
      copy[i] = { ...copy[i], feedback: copy[i].feedback === v ? undefined : v };
      return copy;
    });
    // 피드백은 로컬에만 기록 (제출 없음 — 데모)
    console.log("feedback", { turn: i, value: v });
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-inner">
          <p className="hero-badge">모두의연구소 컨퍼런스</p>
          <h1>모두콘 <span className="accent">ModuCon</span></h1>
          <p className="hero-sub">
            모두의연구소가 여는 연간 개발 컨퍼런스. 연구·개발·빅매치의 결과를 발표하고
            만나는 자리입니다. 궁금한 것은 아래 챗봇에게 — 로컬 모델이 공개 문서에서
            근거를 찾아 답합니다.
          </p>
          <a className="hero-cta" href="#chat">챗봇으로 물어보기 ↓</a>
        </div>
      </header>

      <section className="engine">
        <div className="engine-row">
          <span>답변 엔진:</span>
          <label><input type="radio" checked={engine==="local"} onChange={()=>setEngine("local")} /> 로컬 ollama (qwen3.5:2b)</label>
          <label><input type="radio" checked={engine==="gemini"} onChange={()=>setEngine("gemini")} /> Gemini API</label>
        </div>
        {engine === "gemini" && (
          <div className="engine-row">
            <input
              type="password"
              placeholder="Gemini API 키 (브라우저에만 저장됩니다)"
              value={apiKey}
              onChange={(e) => { setApiKey(e.target.value); localStorage.setItem("gemini_key", e.target.value); }}
            />
          </div>
        )}
      </section>

      {ollamaOk === false && engine === "local" && (
        <div className="banner">
          <strong>로컬 모델(ollama)에 연결할 수 없습니다.</strong>
          <ol>
            <li><code>ollama serve</code> 실행 (또는 Ollama 앱 실행) · 모델 확인: <code>ollama pull qwen3.5:2b</code></li>
            <li>
              github.io에서 열었다면 CORS 허용 — 운영 체제별로 한 번만 설정하고 Ollama를 재시작합니다:
              <div className="os-guide">
                <div><strong>macOS</strong><code>launchctl setenv OLLAMA_ORIGINS "https://*.github.io"</code>입력 후 메뉴 막대의 Ollama 앱을 종료하고 다시 실행합니다.</div>
                <div><strong>Windows</strong>작업 표시줄에서 Ollama를 종료합니다. 설정에서 <code>환경 변수</code>를 검색해 <code>계정의 환경 변수 편집</code>을 열고 새 변수 <code>OLLAMA_ORIGINS</code>에 <code>https://*.github.io</code>를 넣은 뒤 Ollama를 다시 시작합니다.</div>
                <div><strong>Linux</strong><code>sudo systemctl edit ollama.service</code>를 열어 <code>[Service]</code> 아래에 <code>Environment="OLLAMA_ORIGINS=https://*.github.io"</code>를 추가하고 <code>sudo systemctl restart ollama</code>로 재시작합니다.</div>
              </div>
            </li>
          </ol>
          <button onClick={() => pingOllama().then(setOllamaOk)}>다시 확인</button>
        </div>
      )}

      <section className="info">
        <div className="card">
          <h2>모두콘이란?</h2>
          <p>
            모두콘은 모두의연구소가 매년 여는 컨퍼런스로, 함께 만들어가는 개발 문화를
            표방합니다. 자세한 내용은 챗봇에게 "모두콘이 뭐야?"라고 물어보세요.
          </p>
        </div>
        <div className="card">
          <h2>근거 원칙</h2>
          <p>
            이 챗봇의 모든 답변은 공개 문서에서 뽑은 조각에 근거합니다. 자료에 없으면
            없다고 답합니다. 각 답변 아래 출처 칩을 누르면 원문 조각을 볼 수 있습니다.
          </p>
        </div>
        <div className="card">
          <h2>실행 구조</h2>
          <p>
            브라우저가 직접 로컬 ollama 모델(qwen3.5:2b)을 호출하고, 질문 임베딩도
            브라우저에서 실행합니다. 서버·API 키가 없습니다.
          </p>
        </div>
      </section>

      <section id="chat" className="chat">
        <h2>모두콘 안내 챗봇</h2>
        <div className="chat-log">
          {turns.map((t, i) => (
            <div key={i} className={`bubble ${t.role}`}>
              <div className="bubble-text">{t.content || (phase === "stream" && i === turns.length - 1 ? "…" : "")}</div>
              {t.role === "assistant" && t.question && (
                <div className="meta-row">
                  {t.judge ? (
                    <span className={`judge judge-${(t.judge.score ?? 0) >= 70 ? "ok" : "bad"}`}>
                      평가 {t.judge.score}점 · {t.judge.grounded ? "근거 준수" : "근거 이탈"} · {t.judge.noHalluc ? "환각 없음" : "환각 의심"}{t.judge.cited ? " · 출처 표시" : " · 출처 누락"}
                      {t.judge.comment && <em> “{t.judge.comment}”</em>}
                    </span>
                  ) : apiKey ? (
                    <span className="judge">평가 중…</span>
                  ) : null}
                  <span className="feedback">
                    <button aria-label="좋아요" className={t.feedback === "up" ? "on" : ""} onClick={() => setFeedback(i, "up")}>👍</button>
                    <button aria-label="싫어요" className={t.feedback === "down" ? "on" : ""} onClick={() => setFeedback(i, "down")}>👎</button>
                  </span>
                </div>
              )}
              {t.sources && (
                <div className="chips">
                  {t.sources.map((s) => (
                    <button key={s.chunk.id} className="chip" onClick={() => setShowSource(t.sources!)}>
                      {s.chunk.id} · {s.chunk.section} · {(s.score * 100).toFixed(0)}%
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          {phase === "search" && <div className="bubble assistant">근거를 찾고 있습니다…</div>}
          <div ref={bottomRef} />
        </div>
        <div className="chat-input">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
            placeholder="예: 모두콘 참가 방법 알려줘"
            disabled={phase !== "idle"}
          />
          {phase === "stream" ? (
            <button onClick={stop}>정지</button>
          ) : (
            <button onClick={ask} disabled={phase !== "idle" || !input.trim()}>
              보내기
            </button>
          )}
        </div>
      </section>

      {showSource && (
        <div className="modal" onClick={() => setShowSource(null)}>
          <div className="modal-body" onClick={(e) => e.stopPropagation()}>
            <h3>근거 조각</h3>
            {showSource.map((s) => (
              <div key={s.chunk.id} className="source-item">
                <div className="source-meta">
                  {s.chunk.id} · {s.chunk.section} · 유사도 {(s.score * 100).toFixed(0)}%
                </div>
                <p>{s.chunk.text}</p>
                <a href={s.chunk.url} target="_blank" rel="noreferrer">원문 보기 →</a>
              </div>
            ))}
            <button onClick={() => setShowSource(null)}>닫기</button>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>
          모두콘 안내 챗봇 — 로컬 실행 데모. 자료: 모두의연구소 공개 페이지.
          모델: qwen3.5:2b (ollama) · 임베딩: bge-small-en-v1.5 (브라우저).
        </p>
      </footer>
    </div>
  );
}
