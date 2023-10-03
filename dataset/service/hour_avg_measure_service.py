from sqlalchemy import func
from sqlalchemy.orm import Session
from dataset.view.hour_avg_measure import HourAvgMeasure


class HourAvgMeasureService:
    @staticmethod
    def get_last_by_sensor_id(session: Session, sensorId: int):
        return session\
            .query(HourAvgMeasure, func.max(HourAvgMeasure.id))\
            .filter(HourAvgMeasure.sensorId == sensorId)\
            .first()[0]