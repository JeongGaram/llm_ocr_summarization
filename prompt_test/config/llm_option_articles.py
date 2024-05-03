MAP_TEMPLATE = """다음은 뉴스 기사의 일부 내용입니다.
{docs}
주요 내용을 한국어로 요약해주세요. 과도하게 간결한 요약문을 생성하지 말고, 불필요한 내용만을 정제하는 방식으로 긴 요약문을 생성해주세요.
Helpful Answer:"""
REDUCE_TEMPLATE = """다음은 한 뉴스 기사를 분할하여 요약한 요약문의 집합입니다:
{docs}
이 요약문들을 병합 및 요약하여 뉴스 기사의 요약문을 3줄짜리 한국어로 생성해주세요. 기사의 논조가 긍정/부정/중립 중 무엇인지를 분석해주세요. 또한 가장 하단에는 주요 키워드를 해쉬태그(#) 형태로 나열해주세요
Helpful Answer:"""
SMMARIZATION_MAX_TOKEN_NUM = 4000
SMMARIZATION_TOTAL_STOP = 100

GENERATE_INSTRUCTION = "<문서>를 참고하여 <Form>과 동일한 형식의 서론을 작성해주세요"
GENERATE_FORM = "<Form>과학기술정보통신부(장관 , 이하 ‘과기정통부’)는 <내용> 밝혔다</Form>"


CHUNK_SIZE = 1000
CHUNK_OVERLAP_SIZE  = 100
TEMPERATURE = 0.1
TOP_P = 0.95
GENERATE_MAX_TOKEN = 1024

