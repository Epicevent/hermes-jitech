---
name: jitech-weekly-kakaowork-summary
description: 승인된 카카오워크 주간 전건을 근거와 함께 요약합니다.
version: 1.0.0
author: Jitech
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [jitech, kakaowork, weekly-summary, evidence, completeness]
    category: communication
    related_skills: []
---

# Jitech Weekly KakaoWork Summary Skill

승인된 현재 슬롯의 카카오워크 패키지에서 고정된 주간 기간의 전건을
요약한다. 검색 결과나 RAG 상위 문서를 전체 대화로 간주하지 않으며,
`jitech_kakaowork_period_records`가 증명한 범위 밖의 완전성을 주장하지 않는다.

## When to Use

- 사용자가 "최근 일주일", "최근 7일", "지난 7일" 카카오워크 요약을 요청한다.
- 사용자가 "지난주" 카카오워크 전건 또는 주간 업무 요약을 요청한다.
- 모든 대화방을 빠짐없이 보았는지, 누락이 없는지 함께 확인해야 한다.

메일, 임의 날짜 범위, 특정 인물의 장기 이력, 주제 검색에는 이 스킬을
사용하지 않는다. 그런 요청을 주간 전건 요청으로 임의 변환하지 않는다.

## Prerequisites

- `jitech_kakaowork_period_records` 도구가 현재 세션에 노출되어 있어야 한다.
- 현재 슬롯에 `messages.sqlite`와 `membership.json` 카카오 패키지가
  읽기 전용으로 연결되어 있어야 한다.
- 이 스킬은 사용자·방·파일경로·SQL을 입력받지 않는다. 접근범위는 도구가
  현재 슬롯의 membership에서 자동으로 결정한다.

## How to Run

기간 표현을 다음 두 값 중 하나로만 변환한다.

| 사용자 표현 | `period` |
|---|---|
| 최근 일주일, 최근 7일, 지난 7일 | `rolling_7d` |
| 지난주 | `previous_calendar_week` |

다른 기간은 근사하지 말고 지원되는 두 기간 중 어느 것인지 확인한다.

첫 호출은 항상 다음 형태다.

```json
{"operation":"manifest","period":"rolling_7d"}
```

## Quick Reference

| 단계 | operation | 종료 증거 |
|---|---|---|
| 범위 고정 | `manifest` | 절대 기간, 총건수, batch 목록, snapshot token |
| 전건 읽기 | `read_batch` | 모든 batch의 모든 cursor와 최종 coverage digest |
| 완전성 확인 | `reconcile` | `complete`, 처리·실패·미처리 수, 누락·중복 목록 |

주장 근거는 다음 표기를 사용한다.

```text
[카카오워크 | <방 이름> | <local_time> | <stable_message_id>]
```

## Procedure

1. 기간 표현을 위 표의 한 값으로 결정한다.
2. `manifest`를 호출한다.
3. `status=unavailable`이면 검색이나 RAG로 우회하지 않는다. connection의
   diagnostic과 실패 범위를 그대로 알리고 종료한다.
4. freshness가 `stale`이면 이후 기록을 읽을 수는 있지만 "최근 데이터 전체",
   "누락 없음", "전건 완료"라고 표현하지 않는다.
5. 메시지 0건이면 다음을 구분해 종료한다.
   - fresh: "연결됨, 해당 기간 0건"
   - stale: "연결됨, 패키지에서 해당 기간 0건이나 원본 최신성 미확인"
6. manifest의 batch를 순서대로 처리한다. 각 batch는 cursor 없이 시작하고,
   `next_cursor`가 없어질 때까지 직전 값을 그대로 전달한다.
7. 각 page의 모든 record를 읽는다. room, sender, message identity를 다른
   record와 합성하거나 추정하지 않는다.
8. 각 batch에서 다음 항목을 정리한다.
   - 의사결정
   - 요청
   - 일정
   - 문제
   - 수치
   - 미응답 또는 후속 필요
   - 저가치 항목의 종류별 건수
9. 각 항목에는 최소 한 개의 `stable_message_id` 근거를 연결한다. 여러 방의
   같은 이름, 서로 다른 발신자, 서로 다른 행위자를 하나의 사실로 합치지 않는다.
10. 마지막 page에서 받은 `batch_coverage_digest`만 보존한다. 중간 page에는
    batch coverage가 없으며 임의로 계산하거나 추측하지 않는다.
11. 모든 batch 처리 후 다음 형태로 `reconcile`을 호출한다.

```json
{
  "operation": "reconcile",
  "snapshot_token": "<manifest token>",
  "coverage": [
    {"batch_id": "<id>", "coverage_digest": "<final page digest>"}
  ]
}
```

12. `snapshot_mismatch`이면 새 `manifest`부터 전체 절차를 한 번만 재시작한다.
    두 번째 변경은 불완전으로 종료하고 두 번의 변경 사실을 알린다.
13. `complete=true`일 때만 "전건 요약 완료"라고 표시한다.

현재 대표 규모는 단일 증거예산 안이므로 부모 턴에서 순차 처리한다. 향후
manifest의 실제 문자량이 현재 세션의 안전한 증거예산을 넘는 경우에만 기존
`delegate_task`의 batch 실행을 사용한다. 자식에는 해당 batch 조회에 필요한
Kakao 도구만 제공하고, 각 자식은 최종 coverage digest와 근거 ID를 반환해야
한다. 새 runner, DB, daemon, queue를 만들지 않는다.

## Output Format

1. 제목에 실제 절대 시작·종료시각과 기간 종류를 쓴다.
2. 본문에는 의사결정, 요청, 일정, 문제, 핵심 수치를 근거와 함께 쓴다.
3. 부록에는 방별·batch별 메시지 수, 처리 수, 실패 수를 쓴다.
4. 마지막에 다음 완전성 블록을 반드시 쓴다.

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

## Pitfalls

- `kwrag_search` 결과를 전건으로 표시하지 않는다.
- manifest의 총건수만 보고 내용을 요약하지 않는다.
- cursor가 남았는데 다음 batch로 넘어가지 않는다.
- final-page coverage digest가 없는 batch를 완료 처리하지 않는다.
- `failed_messages`를 내용이 확인된 메시지로 취급하지 않는다.
- `stale`, 누락, 중복, digest mismatch를 각주로 숨기지 않는다.
- 설정값이나 예상값을 실제 DB에서 관측된 총건수처럼 쓰지 않는다.
- 근거 ID 없이 사람, 일정, 결정, 수치를 단정하지 않는다.

## Verification

응답 전 다음을 확인한다.

- 절대 기간이 사용자의 "최근 일주일" 또는 "지난주" 의미와 일치한다.
- manifest의 모든 batch가 coverage에 정확히 한 번 존재한다.
- `processed_messages + failed_messages + uncovered_messages`가
  `source_total_messages`와 같다.
- 각 핵심 주장에 실제 record의 stable message ID가 있다.
- `complete=false` 또는 freshness stale이면 완료·전체·누락 없음 표현이 없다.
- 방·발신자·행위자 사이의 오귀속이 없다.
