from typing import List
from sqlalchemy.orm import Session
from dataset.model.experiment import Experiment


class ExperimentsService:
    @staticmethod
    def get_all(session: Session) -> List[Experiment]:
        return session.query(Experiment).all()

    @staticmethod
    def get_by_id(session: Session, experimentId: int) -> Experiment:
        return session.query(Experiment).get(experimentId)

    @staticmethod
    def create(session: Session, experiment: Experiment) -> Experiment:
        session.add(experiment)
        session.commit()
        return experiment

    @staticmethod
    def delete(session: Session, experimentId: int) -> None:
        session.delete(session.query(Experiment).get(experimentId))
        session.commit()
