---
name: jitech-weekly-kakaowork-summary
description: 승인된 카카오워크 주간 전건을 근거와 함께 요약합니다.
version: 1.0.4
author: Jitech
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [jitech, kakaowork, weekly-summary, evidence, completeness]
    category: communication
    related_skills: []
---

# Jitech Weekly KakaoWork Summary

현재 슬롯에 승인된 카카오워크 기록을 기간 전체로 읽고 근거와 완전성을
포함해 요약한다. RAG나 검색 결과를 전건으로 간주하지 않는다.

## 발동 범위

- "최근 일주일", "최근 7일", "지난 7일" → `rolling_7d`
- "지난주" → `previous_calendar_week`
- 메일, 임의 기간, 특정 인물 장기 이력, 주제 검색에는 사용하지 않는다.

## 반드시 이어서 실행

1. `jitech_kakaowork_period_records`의 `manifest`를 호출한다.
2. `status=unavailable`이면 우회 검색 없이 연결 진단과 미확인 범위를 알린다.
3. 0건이면 fresh는 "연결됨, 해당 기간 0건", stale은 최신성 미확인으로 끝낸다.
4. 1건 이상이면 manifest만 보고 답하지 말고 `batches`의 모든 batch ID를
   끝까지 읽는다.
   `jitech_kakaowork_period_records`는 모델 응답 한 번에 정확히 한 번만
   호출한다. 여러 batch나 page의 `read_batch`를 한 응답에 병렬·일괄 호출하지
   않는다. 한 호출의 JSON 결과를 받은 뒤에만 다음 호출을 만든다.
   각 batch는 cursor 없이 `read_batch`를 시작하고 `next_cursor`가 없어질
   때까지 같은 batch를 한 page씩 계속 읽은 뒤 다음 batch로 넘어간다.
   도구 JSON은 그대로 읽으며 `execute_code`, 셸, 파일 도구로 복사하거나
   재파싱하지 않는다. batch 사이에 사용자 확인을 요청하지 않는다.
5. 각 page의 room, sender, local_time, plain_text, `stable_message_id`를
   그대로 사용한다. 서로 다른 방·발신자·행위자를 합치거나 추정하지 않는다.
6. 마지막 page의 `batch_coverage_digest`를 batch별로 한 번만 보존한다.
7. 모든 batch 뒤 `reconcile`을 호출한다.

```json
{"operation":"reconcile","snapshot_ref":"<manifest snapshot_ref>","coverage":[{"batch_id":"<id>","coverage_digest":"<final digest>"}]}
```

`snapshot_mismatch`이면 새 manifest부터 한 번만 다시 시작한다. 두 번째 변경은
불완전으로 종료한다. `complete=true`일 때만 "전건 요약 완료"라고 쓴다.
freshness가 `stale`이거나 `complete=false`이면 전체·누락 없음·완료라고 쓰지 않는다.

## 요약 형식

- 제목: 실제 절대 시작·종료시각과 기간 종류
- 본문: 의사결정, 요청, 일정, 문제, 수치, 미응답/후속 필요
- 각 핵심 주장: `[카카오워크 | 방 | local_time | stable_message_id]`
- 부록: 방·batch별 원본/처리/실패 수
- 마지막 블록:

```text
완전성: 완료 | 불완전
원본 메시지: N
처리 성공: N
처리 실패: N
미처리: N
누락 batch: ...
중복 batch: ...
복호화 실패: N
최신성: fresh | stale
```

응답 전 `processed_messages + failed_messages + uncovered_messages`가
`source_total_messages`와 같은지, manifest의 모든 batch가 coverage에 정확히
한 번 있는지, 모든 핵심 주장에 실제 근거 ID가 있는지 확인한다.
