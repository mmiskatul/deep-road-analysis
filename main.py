#!/usr/bin/env python3
"""
Deep Road Analysis System - Main CLI & Processing Engine
Usage:
    python main.py --source sample_road.jpg
    python main.py --source input_video.mp4 --view
    python main.py --source 0 --view (Webcam mode)
    python main.py --web (Launch interactive Web UI)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any, Optional, Dict
import cv2

from road_analyzer import RoadAnalyzer


def print_rich_dashboard(analytics: dict, elapsed_time: float, output_img_path: str = None, output_json_path: str = None):
    """Prints a styled terminal dashboard using rich if available, else standard text."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich import box

        console = Console()
        summary = analytics["summary"]

        # Header Panel
        status_color = "green"
        if summary["safety_status"] == "CRITICAL ALERT":
            status_color = "bold red"
        elif summary["safety_status"] == "PROXIMITY WARNING":
            status_color = "bold yellow"

        console.print(
            Panel(
                f"[bold cyan]DEEP ROAD ANALYSIS & SCENE PERCEPTION[/bold cyan]\n"
                f"[dim]Inference Time: {elapsed_time:.2f}s | Status: [/dim][{status_color}]{summary['safety_status']}[/{status_color}]",
                box=box.DOUBLE,
                border_style="cyan",
            )
        )

        # Summary KPIs Table
        kpi_table = Table(title="[bold]Scene Overview & Traffic Telemetry[/bold]", box=box.ROUNDED)
        kpi_table.add_column("Metric", style="cyan", no_wrap=True)
        kpi_table.add_column("Value", style="bold white")
        kpi_table.add_column("Assessment / Remarks", style="dim")

        kpi_table.add_row("Total Objects Detected", str(summary["total_objects"]), "All detected road participants")
        kpi_table.add_row("Total Motor Vehicles", str(summary["total_vehicles"]), "Cars, Buses, Trucks, Two-Wheelers")
        kpi_table.add_row("Vulnerable Road Users (VRU)", str(summary["vulnerable_road_users"]), "Pedestrians & Riders")
        kpi_table.add_row("Traffic Signals & Signs", str(summary["infrastructure_elements"]), "Regulatory & Control items")
        kpi_table.add_row(
            "Traffic Density",
            f"{summary['traffic_density']} ({summary['density_index']}%)",
            "Based on vehicle count & road capacity"
        )
        kpi_table.add_row(
            "Road Safety Score",
            f"{summary['road_safety_score']}/100",
            "Computed from proximity hazards & congestion"
        )
        kpi_table.add_row(
            "Active Hazard Alerts",
            f"Critical: {summary['critical_hazards']} | Caution: {summary['caution_hazards']}",
            "Objects in ego-vehicle trajectory"
        )

        console.print(kpi_table)

        # Class Breakdown Table
        breakdown_table = Table(title="[bold]Detected Classes Breakdown[/bold]", box=box.SIMPLE_HEAVY)
        breakdown_table.add_column("Class", style="magenta")
        breakdown_table.add_column("Count", justify="center", style="bold green")

        for cls_name, count in sorted(analytics["counts_by_class"].items(), key=lambda x: x[1], reverse=True):
            breakdown_table.add_row(cls_name.title(), str(count))

        console.print(breakdown_table)

        # Alerts if any
        if analytics["proximity_alerts"]:
            alert_table = Table(title="[bold red]Proximity & Hazard Warnings[/bold red]", box=box.SQUARE)
            alert_table.add_column("Target ID", justify="center")
            alert_table.add_column("Class", style="yellow")
            alert_table.add_column("Severity", justify="center")
            alert_table.add_column("Confidence", justify="center")

            for a in analytics["proximity_alerts"]:
                sev_style = "bold red" if a["hazard_level"] == "CRITICAL" else "yellow"
                alert_table.add_row(
                    str(a["object_id"]),
                    a["class"].title(),
                    f"[{sev_style}]{a['hazard_level']}[/{sev_style}]",
                    f"{int(a['confidence'] * 100)}%"
                )
            console.print(alert_table)

        # Output Links
        if output_img_path or output_json_path:
            out_info = "[bold green]Saved Outputs:[/bold green]\n"
            if output_img_path:
                out_info += f"  - Annotated Visual: [link=file:///{output_img_path}]{output_img_path}[/link]\n"
            if output_json_path:
                out_info += f"  - Analytics Data:   [link=file:///{output_json_path}]{output_json_path}[/link]"
            console.print(Panel(out_info, border_style="green"))

    except ImportError:
        # Fallback to clean standard output
        print("\n" + "=" * 60)
        print("           DEEP ROAD ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Safety Status:         {analytics['summary']['safety_status']}")
        print(f"Road Safety Score:     {analytics['summary']['road_safety_score']}/100")
        print(f"Traffic Density:       {analytics['summary']['traffic_density']}")
        print(f"Total Vehicles:        {analytics['summary']['total_vehicles']}")
        print(f"Vulnerable Users:      {analytics['summary']['vulnerable_road_users']}")
        print(f"Signs & Lights:        {analytics['summary']['infrastructure_elements']}")
        print(f"Critical Hazards:      {analytics['summary']['critical_hazards']}")
        print("-" * 60)
        print("Class Counts:", json.dumps(analytics["counts_by_class"], indent=2))
        if output_img_path:
            print(f"Saved Image: {output_img_path}")
        if output_json_path:
            print(f"Saved JSON:  {output_json_path}")
        print("=" * 60 + "\n")


def process_video(
    analyzer: RoadAnalyzer,
    video_source: Any,
    output_path: str = None,
    output_json_path: str = None,
    conf: float = 0.35,
    show_view: bool = False,
    show_danger_zone: bool = True,
    show_hud: bool = True,
    track: bool = True,
    linkedin_format: bool = False,
    max_frames: int = 0,
):
    """Processes video file or webcam stream with persistent tracking and video-level analytics."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not str(video_source).isdigit() else 0
    total_frames = min(raw_total, max_frames) if (max_frames > 0 and raw_total > 0) else (max_frames if max_frames > 0 else raw_total)

    out_w, out_h = (1080, 1080) if linkedin_format else (width, height)

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Try best codecs for Windows compatibility
        for codec in ["mp4v", "avc1", "XVID"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))
            if temp_writer.isOpened():
                writer = temp_writer
                break
        if writer is None or not writer.isOpened():
            # Fallback to mp4v
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (out_w, out_h))

    frame_idx = 0
    start_total = time.time()
    tracked_unique_ids = set()
    class_totals = {}
    hazard_events_count = 0
    min_safety_score = 100
    peak_vehicles = 0
    peak_density = "FREE FLOW"
    frame_analytics_timeline = []

    # Try rich progress bar if terminal supports it
    use_rich_progress = False
    progress = None
    task_id = None
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        if total_frames > 0:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Analyzing Video[/bold cyan]"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("[dim]Frame {task.completed}/{task.total}[/dim]"),
                TimeRemainingColumn(),
            )
            progress.start()
            task_id = progress.add_task("process", total=total_frames)
            use_rich_progress = True
    except Exception:
        use_rich_progress = False

    if not use_rich_progress:
        print(f"[INFO] Processing video: {video_source} ({total_frames} frames, {fps_in:.1f} FPS)...")
        if show_view:
            print("[INFO] Press 'q' in the window to stop early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            frame_idx += 1
            t_frame_start = time.time()

            # Run analyzer with multi-object tracking enabled
            analytics = analyzer.analyze(frame, conf=conf, track=track)
            annotated = analyzer.render_hud(
                frame,
                analytics,
                show_danger_zone=show_danger_zone,
                show_hud_banner=show_hud,
            )

            dt = max(1e-5, time.time() - t_frame_start)
            fps_proc = 1.0 / dt

            # Draw processing speed in bottom-right corner
            cv2.putText(
                annotated,
                f"AI PROC: {fps_proc:.1f} FPS",
                (width - 170, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

            summary = analytics["summary"]
            # Accumulate metrics
            v_count = summary["total_vehicles"]
            if v_count > peak_vehicles:
                peak_vehicles = v_count
                peak_density = summary["traffic_density"]

            if summary["road_safety_score"] < min_safety_score:
                min_safety_score = summary["road_safety_score"]

            if summary["critical_hazards"] > 0 or summary["caution_hazards"] > 0:
                hazard_events_count += 1

            for det in analytics["detections"]:
                tid = det.get("track_id")
                if tid is not None:
                    tracked_unique_ids.add(tid)
                c_name = det["class"]
                class_totals[c_name] = class_totals.get(c_name, 0) + 1

            # Log sample every 10 frames to timeline
            if frame_idx % 10 == 1 or frame_idx == total_frames:
                frame_analytics_timeline.append({
                    "frame": frame_idx,
                    "timestamp_s": round(frame_idx / fps_in, 2),
                    "vehicles": summary["total_vehicles"],
                    "vulnerable_users": summary["vulnerable_road_users"],
                    "safety_score": summary["road_safety_score"],
                    "density": summary["traffic_density"],
                    "hazards": len(analytics["proximity_alerts"]),
                })

            if linkedin_format:
                display_frame = analyzer.render_linkedin_canvas(annotated, analytics, canvas_size=1080)
            else:
                display_frame = annotated

            if writer:
                writer.write(display_frame)

            if use_rich_progress and progress:
                progress.update(task_id, advance=1)
            elif not use_rich_progress and total_frames > 0 and frame_idx % 25 == 0:
                print(f"  -> Frame {frame_idx}/{total_frames} ({int(frame_idx/total_frames*100)}%) - {fps_proc:.1f} FPS")

            if show_view:
                cv2.imshow("Deep Road Analysis - Video", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User stopped playback.")
                    break
    finally:
        if progress:
            progress.stop()
        cap.release()
        if writer:
            writer.release()
        if show_view:
            cv2.destroyAllWindows()

    total_time = time.time() - start_total
    avg_fps = frame_idx / max(1e-5, total_time)

    # Video summary report dictionary
    video_summary = {
        "video_source": str(video_source),
        "total_frames_analyzed": frame_idx,
        "video_duration_seconds": round(frame_idx / fps_in, 2) if fps_in > 0 else 0,
        "processing_time_seconds": round(total_time, 2),
        "average_fps": round(avg_fps, 2),
        "unique_objects_tracked": len(tracked_unique_ids),
        "peak_vehicles_detected": peak_vehicles,
        "peak_traffic_density": peak_density,
        "hazard_incident_frames": hazard_events_count,
        "minimum_safety_score": min_safety_score,
        "total_class_detections": class_totals,
        "timeline_samples": frame_analytics_timeline,
    }

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(video_summary, f, indent=2)

    # Print rich video summary report
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        console.print(
            Panel(
                f"[bold cyan]VIDEO ANALYSIS COMPLETED[/bold cyan]\n"
                f"[white]Processed {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)[/white]",
                box=box.DOUBLE,
                border_style="green",
            )
        )

        v_table = Table(title="[bold]Video Road Scene Telemetry Summary[/bold]", box=box.ROUNDED)
        v_table.add_column("Metric", style="cyan")
        v_table.add_column("Value", style="bold white")

        v_table.add_row("Total Frames", str(frame_idx))
        v_table.add_row("Video Duration", f"{video_summary['video_duration_seconds']}s")
        v_table.add_row("Average Processing Speed", f"{avg_fps:.1f} FPS")
        v_table.add_row("Unique Objects Tracked", str(len(tracked_unique_ids)))
        v_table.add_row("Peak Vehicles in Single Frame", str(peak_vehicles))
        v_table.add_row("Peak Traffic Density", peak_density)
        v_table.add_row("Hazard Warning Frames", f"{hazard_events_count} frames")
        v_table.add_row("Minimum Road Safety Score", f"{min_safety_score}/100")

        console.print(v_table)

        if output_path or output_json_path:
            out_msg = "[bold green]Saved Outputs:[/bold green]\n"
            if output_path:
                out_msg += f"  - Output Video: [link=file:///{output_path}]{output_path}[/link]\n"
            if output_json_path:
                out_msg += f"  - Video Analytics JSON: [link=file:///{output_json_path}]{output_json_path}[/link]"
            console.print(Panel(out_msg, border_style="green"))

    except ImportError:
        print("\n" + "=" * 60)
        print("         VIDEO ROAD ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Frames Processed:        {frame_idx}")
        print(f"Average FPS:             {avg_fps:.1f}")
        print(f"Unique Tracked Objects:  {len(tracked_unique_ids)}")
        print(f"Peak Vehicles:           {peak_vehicles} ({peak_density})")
        print(f"Hazard Event Frames:     {hazard_events_count}")
        print(f"Min Safety Score:        {min_safety_score}/100")
        if output_path:
            print(f"Saved Video:             {output_path}")
        if output_json_path:
            print(f"Saved Analytics:         {output_json_path}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deep Road Analysis & Scene Perception CLI")
    parser.add_argument("--source", type=str, default=None, help="Input image, video, directory, or webcam index (0)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on ('cpu' or 'cuda')")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save analyzed media and reports")
    parser.add_argument("--no-save", action="store_true", help="Do not save output image/video")
    parser.add_argument("--no-danger-zone", action="store_true", help="Disable ego hazard corridor")
    parser.add_argument("--no-hud", action="store_true", help="Disable top telemetry banner")
    parser.add_argument("--view", action="store_true", help="Display visual preview window")
    parser.add_argument("--export-json", action="store_true", default=True, help="Save structured analytics JSON")
    parser.add_argument("--linkedin", action="store_true", help="Format output for LinkedIn (1:1 1080x1080 with branded title card & live telemetry)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process (0 = all)")
    parser.add_argument("--web", action="store_true", help="Launch interactive Streamlit web dashboard")

    args = parser.parse_args()

    if args.web:
        # Launch Streamlit app
        print("[INFO] Launching interactive Streamlit Web App...")
        import subprocess
        python_exe = sys.executable
        subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"])
        return

    # Check model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file '{args.model}' not found!")
        sys.exit(1)

    # Check and resolve source
    source = args.source
    if not source:
        # Auto-detect best available media
        if os.path.exists("new video .mov"):
            source = "new video .mov"
            print(f"[INFO] No --source specified. Auto-detected video: {source}")
        elif os.path.exists("sample_traffic.mp4"):
            source = "sample_traffic.mp4"
            print(f"[INFO] No --source specified. Auto-detected video: {source}")
        elif os.path.exists("sample_road.jpg"):
            source = "sample_road.jpg"
            print(f"[INFO] No --source specified. Auto-detected image: {source}")
        else:
            print("[ERROR] No input media found! Please specify --source <path>")
            sys.exit(1)

    print(f"[INFO] Initializing Deep Road Analyzer with model: {args.model} on {args.device}...")
    analyzer = RoadAnalyzer(model_path=args.model, device=args.device)

    is_webcam = source.isdigit() or source == "0"
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    os.makedirs(args.output_dir, exist_ok=True)

    if is_webcam:
        cam_idx = int(source)
        out_vid = os.path.join(args.output_dir, "webcam_analyzed.mp4") if not args.no_save else None
        process_video(
            analyzer,
            cam_idx,
            output_path=out_vid,
            conf=args.conf,
            show_view=True,
            show_danger_zone=not args.no_danger_zone,
            show_hud=not args.no_hud,
            linkedin_format=args.linkedin,
            max_frames=args.max_frames,
        )
    elif os.path.isfile(source):
        ext = Path(source).suffix.lower()
        base_name = Path(source).stem

        if ext in video_extensions:
            suffix = "_linkedin.mp4" if args.linkedin else "_analyzed.mp4"
            out_vid = os.path.join(args.output_dir, f"{base_name}{suffix}") if not args.no_save else None
            out_json = os.path.join(args.output_dir, f"{base_name}_video_analytics.json") if args.export_json else None
            process_video(
                analyzer,
                source,
                output_path=out_vid,
                output_json_path=out_json,
                conf=args.conf,
                show_view=args.view,
                show_danger_zone=not args.no_danger_zone,
                show_hud=not args.no_hud,
                track=True,
                linkedin_format=args.linkedin,
                max_frames=args.max_frames,
            )
        elif ext in image_extensions:
            suffix = "_linkedin.jpg" if args.linkedin else "_analyzed.jpg"
            out_img = os.path.join(args.output_dir, f"{base_name}{suffix}") if not args.no_save else None
            out_json = os.path.join(args.output_dir, f"{base_name}_analytics.json") if args.export_json else None

            start = time.time()
            analytics, annotated = analyzer.process_image_file(
                image_path=source,
                output_image_path=None,
                output_json_path=out_json,
                conf=args.conf,
            )
            elapsed = time.time() - start

            if args.linkedin:
                annotated = analyzer.render_linkedin_canvas(annotated, analytics, canvas_size=1080)

            if out_img:
                cv2.imwrite(out_img, annotated)

            print_rich_dashboard(
                analytics=analytics,
                elapsed_time=elapsed,
                output_img_path=out_img,
                output_json_path=out_json,
            )

            if args.view:
                cv2.imshow("Deep Road Analysis", annotated)
                print("[INFO] Press any key in the window to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"[ERROR] Unsupported file format: {ext}")
    elif os.path.isdir(source):
        print(f"[INFO] Batch processing directory: {source}")
        images = [f for f in Path(source).iterdir() if f.suffix.lower() in image_extensions]
        for idx, img_path in enumerate(images, 1):
            out_img = os.path.join(args.output_dir, f"{img_path.stem}_analyzed.jpg") if not args.no_save else None
            out_json = os.path.join(args.output_dir, f"{img_path.stem}_analytics.json") if args.export_json else None
            t0 = time.time()
            analytics, _ = analyzer.process_image_file(
                str(img_path),
                output_image_path=out_img,
                output_json_path=out_json,
                conf=args.conf,
            )
            print(f"[{idx}/{len(images)}] {img_path.name} -> {analytics['summary']['total_objects']} objects | Status: {analytics['summary']['safety_status']} ({time.time()-t0:.2f}s)")
    else:
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
