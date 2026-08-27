// 모두콘 안내 챗봇 — RAG 유틸리티
// 임베딩: embeddinggemma-300m (model_no_gather_q4 변형 — 브라우저 WASM ORT 호환)
//   - transformers.js pipeline()은 q4/q8 기본 파일을 골라 GatherBlockQuantized
//     미지원으로 실패하므로, 토크나이저만 transformers.js로 쓰고
//     ORT 세션은 no_gather_q4 파일로 직접 만든다 (2026-08 헤드리스 검증).
// 검색: moducon-docs.json 정적 벡터스토어와 코사인 유사도 top-k

import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const HF_ONNX = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;
// transformers.js 4.2.0이 쓰는 것과 같은 onnxruntime-web 빌드 (검증된 조합)
const ORT_URL =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort.webgpu.bundle.min.mjs";

interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { dims: number[]; data: Float32Array }>>;
}

export interface DocChunk {
  id: string;
  text: string;
  url: string;
  section: string;
  vector: number[];
}

let session: OrtSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;
let ready: Promise<void> | null = null;

/** 임베딩 모델 내려받기 진행률 (첫 방문 1회, 이후 브라우저 캐시) */
export type EmbedProgress = { pct: number; file: string };
let progressCb: ((p: EmbedProgress) => void) | null = null;
export function onEmbedProgress(cb: (p: EmbedProgress) => void) {
  progressCb = cb;
}

async function fetchWithProgress(url: string, file: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`모델 파일 내려받기 실패 (${res.status}): ${file}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  let lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    const pct = Math.round((got / total) * 100);
    if (pct !== lastPct) {
      lastPct = pct;
      progressCb?.({ pct, file });
    }
  }
  const out = new Uint8Array(got);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/** 토크나이저 + ORT 세션 준비 (첫 호출 시 모델 다운로드, 이후 HTTP 캐시) */
function ensureReady(): Promise<void> {
  if (session && tokenizer) return Promise.resolve();
  if (!ready) {
    ready = (async () => {
      const ort = (await import(/* @vite-ignore */ ORT_URL)) as {
        InferenceSession: { create(
          buf: Uint8Array,
          opts: { executionProviders: string[]; externalData: { path: string; data: Uint8Array }[] },
        ): Promise<OrtSession> };
      };
      const core = await fetchWithProgress(`${HF_ONNX}/model_no_gather_q4.onnx`, "model_no_gather_q4.onnx");
      const data = await fetchWithProgress(`${HF_ONNX}/model_no_gather_q4.onnx_data`, "model_no_gather_q4.onnx_data");
      session = await ort.InferenceSession.create(core, {
        executionProviders: ["wasm"],
        externalData: [{ path: "model_no_gather_q4.onnx_data", data }],
      });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    })().catch((e) => {
      ready = null; // 실패 시 다음 질문에서 재시도 가능
      throw e;
    });
  }
  return ready;
}

/** 문장 → 768차원 벡터 (mean pooling + L2 정규화 — 벡터스토어 생성 방식과 동일) */
export async function embed(text: string): Promise<number[]> {
  await ensureReady();
  const { input_ids, attention_mask } = await tokenizer!(text);
  const out = await session!.run({ input_ids, attention_mask });
  const hs = out.last_hidden_state;
  const [, seq, hid] = hs.dims;
  const am = attention_mask.data as ArrayLike<bigint> | ArrayLike<number>;
  const acc = new Float64Array(hid);
  let cnt = 0;
  for (let s = 0; s < seq; s++) {
    const w = Number(am[s]);
    cnt += w;
    if (!w) continue;
    for (let h = 0; h < hid; h++) acc[h] += hs.data[s * hid + h];
  }
  let norm = 0;
  for (let h = 0; h < hid; h++) {
    acc[h] /= cnt;
    norm += acc[h] * acc[h];
  }
  norm = Math.sqrt(norm);
  const vec = new Array<number>(hid);
  for (let h = 0; h < hid; h++) vec[h] = acc[h] / norm;
  return vec;
}

let corpus: DocChunk[] | null = null;

export async function loadCorpus(): Promise<DocChunk[]> {
  if (corpus) return corpus;
  const res = await fetch(`${import.meta.env.BASE_URL}moducon-docs.json`);
  if (!res.ok) throw new Error(`docs 로드 실패: ${res.status}`);
  corpus = (await res.json()) as DocChunk[];
  return corpus;
}

export interface Retrieved {
  chunk: DocChunk;
  score: number;
}

/** 질문 벡터와 코퍼스 벡터의 내적(=코사인) 상위 k개 */
export async function retrieve(question: string, k = 3): Promise<Retrieved[]> {
  const [docs, q] = await Promise.all([loadCorpus(), embed(question)]);
  const scored = docs.map((chunk) => {
    let dot = 0;
    const v = chunk.vector;
    for (let i = 0; i < v.length; i++) dot += v[i] * q[i];
    return { chunk, score: dot };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/** RAG 시스템 지시 — 근거 원칙을 고정 */
export function buildPrompt(question: string, hits: Retrieved[]): string {
  const context = hits
    .map((h) => `[${h.chunk.id} | ${h.chunk.section}] ${h.chunk.text}`)
    .join("\n\n");
  const best = hits[0]?.score ?? 0;
  const weakNote = best < 0.55
    ? "주의: 검색된 조각의 유사도가 낮습니다. 질문과 완전히 맞는 근거가 아닐 수 있으니, 근거에 있는 내용만 짧게 답하고 자료에 없는 부분은 없다고 말합니다."
    : "자료에 근거한 내용만 답하고, 자료에 없으면 없다고 말합니다.";
  return [
    "다음 자료는 모두의연구소의 컨퍼런스 '모두콘'에 대한 공개 문서에서 뽑은 조각입니다.",
    weakNote,
    "근거가 된 조각의 [ID]를 답 안에서 표시합니다.",
    "",
    "[자료]",
    context,
    "",
    "[질문]",
    question,
  ].join("\n");
}
