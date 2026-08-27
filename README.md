# 모두콘 안내 챗봇 (ModuCon Guide Chatbot)

모두의연구소 컨퍼런스 **모두콘**을 소개하는 페이지 + RAG 챗봇. 서버 없이 브라우저가 로컬 ollama와 직접 통신합니다.

## 실행 (로컬 개발)
```bash
cd app
npm install
npm run dev
```

## 사용자 설정 (로컬 엔진)
1. [ollama 설치](https://ollama.com) 후 모델 받기: `ollama pull qwen3.5:2b`
2. CORS 허용 (github.io에서 열 때):
   ```bash
   launchctl setenv OLLAMA_ORIGINS "https://*.github.io"
   # 이후 Ollama 앱 재시작
   ```
3. 페이지 접속 → 첫 질문 시 임베딩 모델(embeddinggemma-300m, ~200MB) 다운로드 후 캐시

## Gemini 엔진 (선택)
- 상단 라디오에서 "Gemini API" 선택 → API 키 입력 (브라우저 localStorage에만 저장)
- LLM-as-a-Judge 자동 평가도 이 키로 동작 (근거 준수·환각·출처·점수)

## GitHub Pages 배포
```bash
cd app && npm run build
# dist/를 gh-pages 브랜치 또는 Pages 설정의 루트로 게시
```

## 구조
- `docs/` — 모두콘 공개 자료 청크·벡터 (37청크, embeddinggemma 768차원; 2025 본행사 + 역대 회차 2018~2024 + 모두의연구소 소개)
- `app/src/rag.ts` — 질문 임베딩·코사인 검색·프롬프트 조립
- `app/src/ollama.ts` — 로컬 스트리밍 클라이언트
- `app/src/gemini.ts` — Gemini 스트리밍 + LLM-as-a-Judge
- `app/build_docs.py` / `app/embed-docs-browser-path.mjs` — 문서 재구축 스크립트
