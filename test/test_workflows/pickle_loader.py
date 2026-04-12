# code to load the pkl file in debugging to see all its content
# implement arg parser to specify the file path from command line, and if the data should be printed to csv
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(description="Load a pickle file and optionally save its content to CSV.")
    parser.add_argument("--file_path", help="Path to the pickle file to load.", type=str, default="3D-Field-Prediction_3535b822.pkl", required=False)
    parser.add_argument("--to-csv", action="store_true", help="Save the content to a CSV file.", default=True, required=False)
    args = parser.parse_args()

    # Load the pickle file
    with open(args.file_path, "rb") as f:
        data = pickle.load(f)

    # Print the content of the pickle file
    print("Content of the pickle file:")
    print(data)

    # Optionally save to Excel/CSV
    if args.to_csv:
        import json
        import pandas as pd

        # Detect workflow dict (has nodes + edges)
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            xlsx_path = args.file_path.replace(".pkl", ".xlsx")
            _export_workflow_xlsx(data, xlsx_path)
            print(f"Workflow exported to {xlsx_path}")
        else:
            csv_file_path = args.file_path.replace(".pkl", ".csv")
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                row = {k: (json.dumps(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                       for k, v in data.items()}
                df = pd.DataFrame([row])
            else:
                df = pd.DataFrame([{"value": data}])
            df.to_csv(csv_file_path, index=False)
            print(f"Content saved to {csv_file_path}")


def _export_workflow_xlsx(data, path):
    import pandas as pd

    # --- Nodes sheet ---
    node_rows = []
    for node in data.get("nodes", []):
        d = node.get("data", {})
        hp = d.get("hyperparams", {})
        row = {
            "id": node.get("id"),
            "type": node.get("type"),
            "label": d.get("label"),
            "category": d.get("category"),
            # pick whichever role field is present
            "method / model / kind": (
                d.get("method")
                or d.get("model")
                or d.get("inputKind")
                or d.get("validatorKind")
                or d.get("postprocessingKind")
            ),
            "pos_x": node.get("position", {}).get("x"),
            "pos_y": node.get("position", {}).get("y"),
        }
        # flatten hyperparams as hp_* columns
        for k, v in hp.items():
            row[f"hp_{k}"] = v
        node_rows.append(row)
    df_nodes = pd.DataFrame(node_rows)

    # --- Edges sheet ---
    edge_rows = []
    for edge in data.get("edges", []):
        edge_rows.append({
            "id": edge.get("id"),
            "source": edge.get("source"),
            "sourceHandle": edge.get("sourceHandle"),
            "target": edge.get("target"),
            "targetHandle": edge.get("targetHandle"),
        })
    df_edges = pd.DataFrame(edge_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_nodes.to_excel(writer, sheet_name="Nodes", index=False)
        df_edges.to_excel(writer, sheet_name="Edges", index=False)
        # auto-fit column widths
        for sheet_name, df in [("Nodes", df_nodes), ("Edges", df_edges)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, start=1):
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).str.len().max() if not df.empty else 0,
                )
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 50)

if __name__ == "__main__":
    main()