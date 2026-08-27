// 모두콘 안내 챗봇 — RAG 유틸리티
// 임베딩: transformers.js (bge-small-en-v1.5, 브라우저 실행)
// 검색: moducon-docs.json 정적 벡터스토어와 코사인 유사도 top-k

import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

export interface DocChunk {
  id: string;
  text: string;
  url: string;
  section: string;
  vector: number[];
}

let embedder: FeatureExtractionPipeline | null = null;
let embedderReady: Promise<FeatureExtractionPipeline> | null = null;

/** 임베딩 파이프라인 로드 (첫 호출 시 모델 다운로드, 이후 캐시) */
export function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (embedder) return Promise.resolve(embedder);
  if (!embedderReady) {
    embedderReady = pipeline("feature-extraction", "onnx-community/embeddinggemma-300m-ONNX", { dtype: "q4" } as never).then((p) => {
      embedder = p as FeatureExtractionPipeline;
      return embedder;
    });
  }
  return embedderReady;
}

/** 문장 → 384차원 벡터 (정규화 포함 — 코사인 = 내적) */
export async function embed(text: string): Promise<number[]> {
  const ext = await getEmbedder();
  const out = await ext(text, { pooling: "mean", normalize: true });
  return Array.from(out.data as Float32Array);
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
