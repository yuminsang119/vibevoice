"""
주소 → 좌표 변환 (지오코딩) 모듈

ASR로 인식된 주소 텍스트에서 위치 정보를 추출하고,
지오코딩 API를 통해 위경도 좌표로 변환합니다.

지원 API:
    1. 카카오 로컬 API (KAKAO_REST_API_KEY 필요)
    2. 네이버 지도 API (NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 필요)
    3. 국토교통부 도로명주소 API (JUSO_API_KEY 필요)
    4. Nominatim (OpenStreetMap, API 키 불필요 - 폴백용)
"""

import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GeoResult:
    """지오코딩 결과"""
    address: str = ""
    road_address: str = ""
    jibeon_address: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    confidence: float = 0.0
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "road_address": self.road_address,
            "jibeon_address": self.jibeon_address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "confidence": self.confidence,
            "source": self.source,
        }


# ---------- 주소 추출 패턴 ----------

# 도로명주소: ~시/도 ~구/군 ~로/길 번호
ROAD_ADDR_PATTERN = re.compile(
    r"(?P<sido>(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충[북남]|전[북남]|경[북남]|제주)"
    r"(?:특별시|광역시|특별자치시|특별자치도|도)?)\s*"
    r"(?P<sigungu>[가-힣]+[시군구])\s*"
    r"(?P<road>[가-힣0-9]+(?:로|대로|길|번길))\s*"
    r"(?P<number>\d+(?:-\d+)?)"
    r"(?:\s+(?P<detail>[가-힣0-9\s]+(?:동|층|호)))?"
)

# 지번주소: ~시/도 ~구/군 ~동/리 번지
JIBEON_ADDR_PATTERN = re.compile(
    r"(?P<sido>(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충[북남]|전[북남]|경[북남]|제주)"
    r"(?:특별시|광역시|특별자치시|특별자치도|도)?)\s*"
    r"(?P<sigungu>[가-힣]+[시군구])\s*"
    r"(?P<dong>[가-힣]+[동리읍면])\s*"
    r"(?P<number>\d+(?:-\d+)?)\s*번?지?"
)

# 랜드마크 기반 위치: ~역 N번 출구, ~건물 앞/뒤
LANDMARK_PATTERN = re.compile(
    r"(?P<landmark>[가-힣A-Za-z0-9]+(?:역|센터|타워|빌딩|아파트|학교|병원|공원|시장|마트|백화점))"
    r"(?:\s*(?P<exit>\d+번\s*출구))?"
    r"(?:\s*(?:에서|부터|의))?"
    r"(?:\s*(?P<direction>앞|뒤|옆|근처|건너편|맞은편|동쪽|서쪽|남쪽|북쪽))?"
)


def extract_addresses(text: str) -> List[Dict[str, str]]:
    """
    텍스트에서 주소 패턴을 추출합니다.

    Returns:
        주소 정보 딕셔너리 리스트
        [{"type": "road"|"jibeon"|"landmark", "address": "...", "detail": "..."}]
    """
    results = []

    # 도로명주소 추출
    for m in ROAD_ADDR_PATTERN.finditer(text):
        addr = f"{m.group('sido')} {m.group('sigungu')} {m.group('road')} {m.group('number')}"
        detail = m.group("detail") or ""
        results.append({
            "type": "road",
            "address": addr.strip(),
            "detail": detail.strip(),
            "full": f"{addr} {detail}".strip(),
        })

    # 지번주소 추출
    for m in JIBEON_ADDR_PATTERN.finditer(text):
        addr = f"{m.group('sido')} {m.group('sigungu')} {m.group('dong')} {m.group('number')}"
        results.append({
            "type": "jibeon",
            "address": addr.strip(),
            "detail": "",
            "full": addr.strip(),
        })

    # 랜드마크 추출 (주소가 없을 때만)
    if not results:
        for m in LANDMARK_PATTERN.finditer(text):
            landmark = m.group("landmark")
            exit_info = m.group("exit") or ""
            direction = m.group("direction") or ""
            full = f"{landmark} {exit_info} {direction}".strip()
            results.append({
                "type": "landmark",
                "address": landmark,
                "detail": f"{exit_info} {direction}".strip(),
                "full": full,
            })

    return results


# ---------- 지오코딩 API ----------

async def _geocode_kakao(address: str) -> Optional[GeoResult]:
    """카카오 로컬 API로 지오코딩"""
    import httpx

    api_key = os.environ.get("KAKAO_REST_API_KEY")
    if not api_key:
        return None

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address, "analyze_type": "similar"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        docs = data.get("documents", [])
        if not docs:
            # 키워드 검색 폴백
            url = "https://dapi.kakao.com/v2/local/search/keyword.json"
            params = {"query": address}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
            docs = data.get("documents", [])

        if not docs:
            return None

        doc = docs[0]
        road = doc.get("road_address") or {}
        jibeon = doc.get("address") or {}

        return GeoResult(
            address=address,
            road_address=road.get("address_name", doc.get("address_name", "")),
            jibeon_address=jibeon.get("address_name", ""),
            latitude=float(doc.get("y", 0)),
            longitude=float(doc.get("x", 0)),
            confidence=0.9,
            source="kakao",
            raw=doc,
        )
    except Exception as e:
        logger.warning(f"카카오 지오코딩 실패: {e}")
        return None


async def _geocode_naver(address: str) -> Optional[GeoResult]:
    """네이버 지도 API로 지오코딩"""
    import httpx

    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }
    params = {"query": address}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        addrs = data.get("addresses", [])
        if not addrs:
            return None

        addr = addrs[0]
        return GeoResult(
            address=address,
            road_address=addr.get("roadAddress", ""),
            jibeon_address=addr.get("jibunAddress", ""),
            latitude=float(addr.get("y", 0)),
            longitude=float(addr.get("x", 0)),
            confidence=0.85,
            source="naver",
            raw=addr,
        )
    except Exception as e:
        logger.warning(f"네이버 지오코딩 실패: {e}")
        return None


async def _geocode_juso(address: str) -> Optional[GeoResult]:
    """국토교통부 도로명주소 API로 지오코딩"""
    import httpx

    api_key = os.environ.get("JUSO_API_KEY")
    if not api_key:
        return None

    # 1단계: 주소 검색
    url = "https://business.juso.go.kr/addrlink/addrLinkApi.do"
    params = {
        "confmKey": api_key,
        "keyword": address,
        "resultType": "json",
        "countPerPage": "1",
        "currentPage": "1",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", {}).get("juso", [])
        if not results:
            return None

        juso = results[0]
        road_addr = juso.get("roadAddr", "")
        jibeon_addr = juso.get("jibunAddr", "")

        # 2단계: 좌표 변환
        coord_url = "https://business.juso.go.kr/addrlink/addrCoordApi.do"
        coord_params = {
            "confmKey": api_key,
            "admCd": juso.get("admCd", ""),
            "rnMgtSn": juso.get("rnMgtSn", ""),
            "udrtYn": juso.get("udrtYn", "0"),
            "buldMnnm": juso.get("buldMnnm", ""),
            "buldSlno": juso.get("buldSlno", ""),
            "resultType": "json",
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            coord_resp = await client.get(coord_url, params=coord_params)
            coord_resp.raise_for_status()
            coord_data = coord_resp.json()

        coords = coord_data.get("results", {}).get("juso", [])
        if coords:
            lat = float(coords[0].get("entY", 0))
            lng = float(coords[0].get("entX", 0))
        else:
            lat, lng = 0.0, 0.0

        return GeoResult(
            address=address,
            road_address=road_addr,
            jibeon_address=jibeon_addr,
            latitude=lat,
            longitude=lng,
            confidence=0.95,
            source="juso",
            raw=juso,
        )
    except Exception as e:
        logger.warning(f"국토교통부 지오코딩 실패: {e}")
        return None


async def _geocode_nominatim(address: str) -> Optional[GeoResult]:
    """OpenStreetMap Nominatim으로 지오코딩 (폴백)"""
    import httpx

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": "1",
        "countrycodes": "kr",
        "accept-language": "ko",
    }
    headers = {"User-Agent": "VibeVoice-119-ASR/1.0"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not data:
            return None

        result = data[0]
        return GeoResult(
            address=address,
            road_address=result.get("display_name", ""),
            jibeon_address="",
            latitude=float(result.get("lat", 0)),
            longitude=float(result.get("lon", 0)),
            confidence=0.6,
            source="nominatim",
            raw=result,
        )
    except Exception as e:
        logger.warning(f"Nominatim 지오코딩 실패: {e}")
        return None


async def geocode(address: str) -> Optional[GeoResult]:
    """
    주소를 좌표로 변환합니다.
    카카오 → 네이버 → 국토교통부 → Nominatim 순으로 시도합니다.
    """
    for fn in [_geocode_kakao, _geocode_naver, _geocode_juso, _geocode_nominatim]:
        result = await fn(address)
        if result and result.latitude != 0.0:
            logger.info(
                f"지오코딩 성공 [{result.source}]: {address} → "
                f"({result.latitude:.6f}, {result.longitude:.6f})"
            )
            return result
    logger.warning(f"지오코딩 실패: {address}")
    return None


async def geocode_from_segments(segments: List[Dict]) -> List[Dict[str, Any]]:
    """
    ASR 전사 결과(segments)에서 주소를 추출하고 좌표로 변환합니다.

    Returns:
        [{"extracted": {...}, "geocode": {...}, "map_url": "..."}]
    """
    results = []
    full_text = " ".join(
        seg.get("Content", seg.get("text", "")) for seg in segments
    )

    addresses = extract_addresses(full_text)
    seen = set()

    for addr_info in addresses:
        addr_key = addr_info["address"]
        if addr_key in seen:
            continue
        seen.add(addr_key)

        geo = await geocode(addr_info["full"])
        entry = {
            "extracted": addr_info,
            "geocode": geo.to_dict() if geo else None,
            "map_urls": {},
        }

        if geo and geo.latitude != 0.0:
            entry["map_urls"] = {
                "kakao": (
                    f"https://map.kakao.com/link/map/{urllib.parse.quote(geo.road_address or addr_key)},"
                    f"{geo.latitude},{geo.longitude}"
                ),
                "naver": (
                    f"https://map.naver.com/v5/search/{urllib.parse.quote(geo.road_address or addr_key)}"
                    f"?c={geo.longitude},{geo.latitude},15,0,0,0,dh"
                ),
                "google": (
                    f"https://www.google.com/maps/search/?api=1"
                    f"&query={geo.latitude},{geo.longitude}"
                ),
            }

        results.append(entry)

    return results
