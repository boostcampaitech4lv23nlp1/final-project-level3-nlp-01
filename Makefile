run_server:
	python3 backend.py

run_client:
	python3 -m streamlit run frontend.py --server.fileWatcherType None --server.port=30004

run_app: run_server run_client