# PRD — 모두콘 안내 챗봇 (ModuCon Guide Chatbot)

## 1. 개요
모두의연구소의 연간 개발 컨퍼런스 **모두콘**을 소개하는 정적 웹 페이지 + RAG 챗봇. 사용자의 브라우저가 로컬 ollama(qwen3.5:2b)와 직접 통신하며, 임베딩도 브라우저에서 실행한다. 서버리스·API 키 불필요.

## 2. 목표 / 비목표
**목표**
- 모두콘 소개 페이지: 무엇인지·일정·참가 방법·프로그램 구성 안내
- RAG 챗봇: 모두콘 공개 자료 기반으로 답변, 출처 표시, 스트리밍 출력
- LMS day-31 튜토리얼 산출물: 4일 교육(지시·맥락·도구·평가)의 종합 실습

**비목표 (v1)**
- 로그인·사용자 관리, 발표 논문 전문·시간표 DB 전체 검색, 다국어, 모바일 최적화(동작은 해야 함)

## 3. 아키텍처
```
GitHub Pages (정적)
 └─ React + Vite SPA
     ├─ 페이지: 모두콘 소개 (히어로·프로그램·FAQ)
     ├─ 챗봇 패널
     │   ├─ 임베딩: transformers.js (bge-small-en-v1.5, WebGPU/WASM)
     │   ├─ 벡터스토어: moducon-docs.json (청크+벡터, 코사인 유사도 top-k)
     │   └─ 답변: fetch http://localhost:11434/api/chat (stream:true)
     └─ 설정: OLLAMA_ORIGINS 가이드 배너
```
- 백엔드 없음. 빌드 산출물(dist/)을 gh-pages로 배포.
- 청킹·임베딩은 **사전 오프라인 단계**(노트북에서 python)로 수행, 결과 JSON을 정적 파일로 포함. 런타임에 임베딩되는 것은 **사용자 질문**뿐(질의 임베딩 = bge-small로 브라우저 실행).

## 4. 기능 명세
### 4.1 소개 페이지
- 히어로: 모두콘 슬로건·최근 대회 연도
- 모두콘이란 / 참가 방법 / 프로그램(발표·부스·네트워킹) / FAQ 아코디언
- 자료 출처 표기 (수집 페이지 목록)

### 4.2 챗봇
- 채팅 UI: 사용자/봇 버블, 마크다운 렌더, 출처 칩(조각 id 클릭 → 원문 하이라이트)
- 스트리밍: ollama `/api/chat` `stream:true` → ReadableStream, 토큰 단위 append
- RAG 파이프라인 (브라우저):
  1) 질문 임베딩(transformers.js)
  2) moducon-docs.json 벡터와 코사인 유사도 → top-3
  3) 시스템 지시 + "주어진 자료 근거로만 답하고 없으면 모른다고" + 조각 3개
  4) 스트리밍 답변 + 출처 표시
- 상태 표시: 임베딩 모델 로딩(첫 방문 ~30MB) / ollama 미실행 감지 → 설정 안내 배너
- 실패 처리: ollama 403(CORS)·연결 거부 → "OLLAMA_ORIGINS 설정법" 안내 카드

### 4.3 데이터 (사전 구축)
- 소스: 모두의연구소 홈·모두콘 소개/후기 공개 페이지 (수집)
- 청크: 제목+문단 단위, 200~600자, 메타(출처 URL·섹션)
- 임베딩: BAAI/bge-small-en-v1.5 — 오프라인(fastembed)과 브라우저(transformers.js)가 같은 모델이라 벡터 공간 호환
- 산출: `public/moducon-docs.json` `{id, text, url, section, vector[384]}`

## 5. 기술 스택
- React 18 + Vite + TypeScript
- @huggingface/transformers (transformers.js v3, WebGPU 폴백 WASM)
- 스타일: 순수 CSS (의존 최소)
- 배포: GitHub Actions → gh-pages

## 6. 사용자 설정 (문서 + UI 안내)
1. ollama 설치·qwen3.5:2b pull
2. `launchctl setenv OLLAMA_ORIGINS "https://*.github.io"` 후 Ollama 재시작
3. 페이지 접속 → 첫 회 임베딩 모델 다운로드

## 7. 성능·제한
- 첫 질의: 임베딩 모델 로드 5~15초(이후 브라우저 캐시)
- 스트리밍 latency: 로컬 qwen3.5:2b 토큰당 ~50ms
- 청크 수 수십~수백 수준 → 전체 로드 코사인 계산으로 충분

## 8. 수용 기준 (Acceptance)
- [ ] 페이지에 모두콘 소개가 출처와 함께 표시된다
- [ ] "모두콘이 뭐야?" 질문에 근거 조각을 인용해 스트리밍 답변
- [ ] 자료에 없는 질문("내일 날씨")에 "자료에 없다"고 답한다
- [ ] 출처 칩 클릭 → 원문 표시
- [ ] ollama 꺼진 상태 → 안내 배너 표시, 재시도 동작
- [ ] github.io에서 정상 동작 (CORS 설정 후)

## 9. 일정 (오늘)
1. 자료 수집·청킹·JSON 구축 (python)
2. Vite 스캐폴드 + 소개 페이지
3. 챗봇: 임베딩 로더 + 검색 + 스트리밍 UI
4. 로컬 통합 실측 → CORS 실측 → github pages 배포
5. README (설정 가이드)
