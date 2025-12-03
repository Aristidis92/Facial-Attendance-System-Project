from datetime import timedelta, date
from attendance.models import Attendance
from django.db.models import Count

def get_summary(student, timeframe='weekly'):
    today = date.today()
    start = today - timedelta(days=7) if timeframe == 'weekly' else today - timedelta(days=30)
    summary = Attendance.objects.filter(
        student=student,
        date__range=(start, today)
    ).values('status').annotate(count=Count('status'))

    return {item['status']: item['count'] for item in summary}
