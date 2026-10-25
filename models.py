from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SPLITS = ("train", "val", "test", "unassigned")


class Base(DeclarativeBase):
    pass


class Lesion(Base):

    __tablename__ = "lesions"
    __table_args__ = (
        UniqueConstraint("source_dataset", "external_lesion_id"),
        CheckConstraint("split IN ('train','val','test','unassigned')", name="ck_lesion_split"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_dataset: Mapped[str] = mapped_column(String(50))
    external_lesion_id: Mapped[str] = mapped_column(String(100))
    split: Mapped[str] = mapped_column(String(12), default="unassigned", index=True)
    age: Mapped[Optional[float]] = mapped_column(Float)
    sex: Mapped[Optional[str]] = mapped_column(String(20))
    localization: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    images: Mapped[list["ImageRecord"]] = relationship(back_populates="lesion")


class ImageRecord(Base):
    __tablename__ = "images"
    __table_args__ = (
        CheckConstraint("label_source IN ('dataset','expert','none')", name="ck_image_label_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lesion_id: Mapped[int] = mapped_column(ForeignKey("lesions.id"), index=True)
    external_image_id: Mapped[Optional[str]] = mapped_column(String(100))
    file_path: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(50))
    dx_label: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    dx_type: Mapped[Optional[str]] = mapped_column(String(30))
    label_source: Mapped[str] = mapped_column(String(20), default="none")
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lesion: Mapped[Lesion] = relationship(back_populates="images")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    architecture: Mapped[Optional[str]] = mapped_column(String(100))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    weights_path: Mapped[str] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    probs: Mapped[dict] = mapped_column(JSONB)
    top_class: Mapped[str] = mapped_column(String(20))
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    image: Mapped[ImageRecord] = relationship(back_populates="predictions")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="prediction")


class Feedback(Base):

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    confirmed_label: Mapped[str] = mapped_column(String(20))
    confirmed_by: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped[Prediction] = relationship(back_populates="feedback")