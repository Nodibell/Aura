import argparse
import json
import sys
import os
import pandas as pd

def run_query(db_type, query, conn_params, output_csv):
    conn = None
    try:
        # SQLite
        if db_type == "sqlite":
            import sqlite3
            db_path = conn_params.get("db_path")
            if not db_path:
                raise ValueError("SQLite requires a 'db_path' parameter.")
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_csv, index=False)
            print(json.dumps({
                "success": True, 
                "row_count": len(df), 
                "columns": list(df.columns)
            }))
            return

        # PostgreSQL
        elif db_type == "postgresql":
            try:
                import psycopg2
            except ImportError:
                raise ImportError("PostgreSQL client library 'psycopg2-binary' is not installed. Run '.venv/bin/pip install psycopg2-binary' to install it.")
            
            port = int(conn_params.get("port", 5432))
            conn = psycopg2.connect(
                host=conn_params.get("host"),
                port=port,
                database=conn_params.get("database"),
                user=conn_params.get("user"),
                password=conn_params.get("password"),
                connect_timeout=10
            )
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_csv, index=False)
            print(json.dumps({
                "success": True, 
                "row_count": len(df), 
                "columns": list(df.columns)
            }))
            return

        # MySQL
        elif db_type == "mysql":
            try:
                import pymysql
            except ImportError:
                raise ImportError("MySQL client library 'pymysql' is not installed. Run '.venv/bin/pip install pymysql' to install it.")
            
            port = int(conn_params.get("port", 3306))
            conn = pymysql.connect(
                host=conn_params.get("host"),
                port=port,
                database=conn_params.get("database"),
                user=conn_params.get("user"),
                password=conn_params.get("password"),
                connect_timeout=10
            )
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_csv, index=False)
            print(json.dumps({
                "success": True, 
                "row_count": len(df), 
                "columns": list(df.columns)
            }))
            return

        # Google BigQuery
        elif db_type == "bigquery":
            try:
                from google.cloud import bigquery
            except ImportError:
                raise ImportError("Google BigQuery client library 'google-cloud-bigquery' is not installed. Run '.venv/bin/pip install google-cloud-bigquery pandas-gbq' to install it.")
            
            creds_path = conn_params.get("credentials_path")
            if creds_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            
            project = conn_params.get("project")
            client = bigquery.Client(project=project)
            query_job = client.query(query)
            df = query_job.to_dataframe()
            df.to_csv(output_csv, index=False)
            print(json.dumps({
                "success": True, 
                "row_count": len(df), 
                "columns": list(df.columns)
            }))
            return

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aura Database Ingestion Runner")
    parser.add_argument("--db-type", required=True, choices=["sqlite", "postgresql", "mysql", "bigquery"], help="Database type")
    parser.add_argument("--query", required=True, help="SQL query to execute")
    parser.add_argument("--conn-params", required=True, help="JSON connection parameters")
    parser.add_argument("--output-csv", required=True, help="Output destination CSV path")

    args = parser.parse_args()

    try:
        conn_params = json.loads(args.conn_params)
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Invalid connection parameters JSON: {str(e)}"
        }))
        sys.exit(1)

    run_query(args.db_type, args.query, conn_params, args.output_csv)
