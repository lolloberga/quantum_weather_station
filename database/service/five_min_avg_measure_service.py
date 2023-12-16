from sqlalchemy.orm import Session
from database.view.five_min_avg_measure import FiveMinAvgMeasure
from sqlalchemy import func


class FiveMinAvgMeasureService:
    @staticmethod
    def get_by_sensor_id_and_date_between(session: Session, sensorId: int, startDate, endDate):
        return session.query(FiveMinAvgMeasure)\
            .with_entities(FiveMinAvgMeasure.date, FiveMinAvgMeasure.hour, FiveMinAvgMeasure.minute, FiveMinAvgMeasure.value)\
            .filter(FiveMinAvgMeasure.sensorId == sensorId)\
            .filter(FiveMinAvgMeasure.date >= startDate)\
            .filter(FiveMinAvgMeasure.date <= endDate)\
            .all()

    @staticmethod
    def get_last_by_sensor_id(session: Session, sensorId: int):
        return session\
            .query(FiveMinAvgMeasure, func.max(FiveMinAvgMeasure.id))\
            .filter(FiveMinAvgMeasure.sensorId == sensorId)\
            .first()[0]

    @staticmethod
    def get_by_sensor_id_and_timestamp(session: Session, sensorId: int, date, hour, minute):
        return session.query(FiveMinAvgMeasure) \
            .filter(FiveMinAvgMeasure.sensorId == sensorId) \
            .filter(FiveMinAvgMeasure.date == date) \
            .filter(FiveMinAvgMeasure.hour == hour) \
            .filter(FiveMinAvgMeasure.minute == minute) \
            .first()