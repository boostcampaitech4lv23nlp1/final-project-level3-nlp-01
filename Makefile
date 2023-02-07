run_server:
	python3 -m app

run_client:
	python3 -m streamlit run app/client.py --server.fileWatcherType None --server.port=30001

run_app: run_server run_client