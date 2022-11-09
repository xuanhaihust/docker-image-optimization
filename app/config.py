import os

PREFIX = "PLATFORM_OCR_"

REDIS_URI = os.getenv(PREFIX + "REDIS_URI") or 'redis://:admin@localhost:6379'

#
# serving
#
SERVING_HOST = os.getenv(PREFIX + "SERVING_HOST") or "localhost:8001"
SERVING_TRITON_PROTOCOL = os.getenv(
    PREFIX + "SERVING_TRITON_PROTOCOL") or "grpc"

#
# torchserve
#
TORCHSERVE_INFERENCE_URL = os.getenv(
    PREFIX + "TORCHSERVE_INFERENCE_URL") or "http://localhost:8080"
TORCHSERVE_MANAGEMENT_URL = os.getenv(
    PREFIX + "TORCHSERVE_MANAGEMENT_URL") or "http://localhost:8081"

# classify document
KNOWN_DOCUMENT_TYPES = [
    'giay_khai_sinh', 'giay_ra_vien', 'dang_ky_ket_hon', 'hoa_don', 'don_thuoc',
    'chung_minh_quan_doi', 'cmnd', 'passport', 'dang_ky_xe'
]