from db import SessionLocal, engine
from models import Base, ModelVersion


def seed():
    # 1. Automatically build database tables if they do not exist
    print("Verifying database tables exist...")
    Base.metadata.create_all(bind=engine)

    # 2. Add the active model record mapping
    with SessionLocal() as db:
        active_model = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()

        if not active_model:
            print("No active model found in the database. Creating relative reference...")
            prod_model = ModelVersion(
                name="v1-production",
                weights_path="models/model.pt",
                is_active=True,
                metrics={"classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]},
            )
            db.add(prod_model)
            db.commit()
            print("Database setup and model tracking initialized successfully!")
        else:
            # Overwrite if the entry contains a Windows path leak
            if "C:\\" in active_model.weights_path or "PycharmProjects" in active_model.weights_path:
                print(f"Correcting Windows path reference entry: '{active_model.weights_path}'")
                active_model.weights_path = "models/model.pt"
                db.commit()
                print("Database record successfully synchronized to a relative Linux structure.")
            else:
                print(f"Active model record '{active_model.name}' already correctly configured.")


if __name__ == "__main__":
    seed()
