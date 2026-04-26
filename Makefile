.PHONY: install run clean lint

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf src/assets/temp/* 2>/dev/null || true

lint:
	python -m flake8 src/ app.py --max-line-length=120 --ignore=E501,W503

validate:
	python -c "from src.helpers.config import get_settings; s=get_settings(); print('✅ Config OK:', s.APP_NAME, s.APP_VERSION)"
