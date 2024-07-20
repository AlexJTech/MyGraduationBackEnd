from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from fileProc.models import HdrFile, ProcRecord, DatFile

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_statistics(request):
    user = request.user
    hdr_file_count = HdrFile.objects.filter(user=user).count()
    proc_record_count = ProcRecord.objects.filter(user=user).count()
    dat_file_count = DatFile.objects.filter(user=user).count()
    hdr_and_dat_file_count = hdr_file_count + dat_file_count

    return Response({
        'hdr_file_count': hdr_file_count,
        'proc_record_count': proc_record_count,
        'hdr_and_dat_file_count': hdr_and_dat_file_count,
    })
