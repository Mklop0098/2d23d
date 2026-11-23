import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import logging
from pathlib import Path
from html import escape
from urllib.parse import quote, urlencode
import gradio as gr
from engine import ROOT, QUALITY, SHAPE_MODELS, SPOT_MODES, HALFMESH_MODES, SMOOTHING_METHODS, generate

VIEW_FILES = {'Đất sét': 'geometry.glb', 'Có màu': 'model.glb',
              'Spot trước sửa': 'spot-before.glb', 'Spot sau sửa': 'spot-after.glb',
              'MeshLab trước xử lý': 'meshlab-before.glb', 'MeshLab sau xử lý': 'meshlab-after.glb'}


def viewer_html(path=None, clay=False):
    source = '/viewer/index.html'
    if path:
        source += '?' + urlencode({'model': '/file=' + quote(str(Path(path).resolve()).replace('\\', '/'), safe='/:'),
                                    'clay': int(clay),
                                    'spot': int(Path(path).name in ('spot-before.glb', 'spot-after.glb'))})
    return (f'<iframe src="{escape(source, quote=True)}" title="Xem mô hình bằng Three.js" '
            'style="width:100%;height:650px;border:0;border-radius:10px" allow="fullscreen"></iframe>')


def convert(image, remove_background, shape_model, quality, texture, threshold, relief, smooth,
            photo_texture, seed, normal_detail, view_mode, mesh_cleanup=True,
            spot_mode=SPOT_MODES[1], spot_threshold=3.5, spot_max_displacement=0.25,
            halfmesh_mode=HALFMESH_MODES[1], max_hole_edges=12,
            smoothing_method=SMOOTHING_METHODS[1], meshlab_angle=45.0, meshlab_iterations=1,
            measure_deviation=True,
            progress=gr.Progress()):
    def report(message):
        if message.endswith('%'):
            progress((int(message.rsplit(' ', 1)[-1][:-1]), 100), desc='Đang lấy mẫu hình khối')
        else:
            progress((0, None), desc=message)
    try:
        folder, metrics = generate(image, remove_background, quality, report,
                                   texture=texture, threshold=threshold, relief=relief,
                                   smooth=smooth, photo_texture=photo_texture,
                                   shape_model=shape_model, seed=seed, normal_detail=normal_detail,
                                   mesh_cleanup=mesh_cleanup, spot_mode=spot_mode,
                                   spot_threshold=spot_threshold, spot_max_displacement=spot_max_displacement,
                                   halfmesh_mode=halfmesh_mode, max_hole_edges=max_hole_edges,
                                   smoothing_method=smoothing_method, meshlab_angle=meshlab_angle,
                                   meshlab_iterations=meshlab_iterations, measure_deviation=measure_deviation)
    except Exception as error:
        logging.exception('Không thể tạo mô hình')
        raise gr.Error(f'Không thể tạo mô hình. {error}') from error
    status = (f"{metrics['shape_model']} · Hoàn thành trong {metrics['seconds']:.1f} giây · "
              f"RAM tiến trình cao nhất {metrics['peak_process_ram_gib']:.2f} GiB · "
              f"{metrics['faces']:,} mặt lưới · Lưới {metrics['resolution']} · "
              f"Texture {metrics['texture_resolution']} px")
    topology = metrics.get('halfmesh')
    if topology:
        status += f" · halfmesh {topology['faces_before']:,} → {topology['faces_after']:,} mặt, vá {topology['holes_closed']} lỗ"
    meshlab = metrics.get('meshlab')
    if meshlab and meshlab['measurement']:
        deviation = meshlab['measurement']['max_percent_bbox_diagonal']
        status += f' · Độ lệch bề mặt ước lượng tối đa {deviation:.3f}% đường chéo khung bao'
    spot = metrics.get('spot')
    if spot:
        if spot['status'] == 'skipped':
            status += ' · Spot đã bỏ qua vì lưới không phù hợp, xem spot-report.json'
        else:
            status += f" · Spot phát hiện {spot['n_flagged']} điểm, đã dịch chuyển {spot['n_moved']} điểm"
        if spot.get('high_mesh', {}).get('status') == 'skipped':
            status += ' · Spot đã bỏ qua lưới bake normal, xem spot-high-report.json'
    preview = folder / VIEW_FILES[view_mode]
    if not preview.is_file():
        preview = folder / 'geometry.glb'
        status += ' · Chế độ so sánh cần bật bước xử lý tương ứng khi tạo mô hình'
    files = [str(folder / name) for name in ('model.glb', 'model.obj', 'halfmesh-report.json', 'meshlab-report.json',
                                            'spot-report.json', 'spot-high-report.json')
             if (folder / name).is_file()]
    return viewer_html(preview), files, status, str(folder / 'input.png'), str(folder)


def change_view(view_mode, folder):
    if not folder:
        return gr.skip()
    if Path(folder).suffix.lower() == '.glb':
        if view_mode not in ('Đất sét', 'Có màu'):
            raise gr.Error('Chế độ so sánh cần kết quả tạo mới có bật bước xử lý tương ứng.')
        return viewer_html(folder, clay=view_mode == 'Đất sét')
    preview = Path(folder) / VIEW_FILES[view_mode]
    if not preview.is_file():
        raise gr.Error('Kết quả này chưa có bản xem tương ứng. Hãy bật bước xử lý và tạo lại mô hình.')
    return viewer_html(preview)


def open_glb(path):
    if not path:
        return gr.skip(), gr.skip(), gr.skip()
    return viewer_html(path), str(path), 'Có màu'


with gr.Blocks(title='Ảnh thành 3D', analytics_enabled=False) as demo:
    gr.Markdown('# Ảnh thành 3D\nChọn ảnh một vật thể rõ nét. Kết quả được lưu trên máy.')
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='filepath', sources=['upload'], label='Ảnh JPG hoặc PNG')
            remove_background = gr.Checkbox(value=True, label='Tách nền tự động')
            shape_model = gr.Radio(list(SHAPE_MODELS), value=SHAPE_MODELS[0], label='Mô hình hình khối',
                                   info='Hunyuan mini dựng hông, lưng, bệ đúng hơn (~2 phút, RAM ~14 GB). '
                                        'TripoSR nhanh hơn (~1 phút) nhưng phần khuất kém.')
            quality = gr.Radio(list(QUALITY), value='Chi tiết', label='Chất lượng hình khối',
                               info='TripoSR: lưới 192 / 256 / 384. Hunyuan: lưới 192 / 256 / 320.')
            relief = gr.Slider(0, 2, value=1.5, step=0.1, label='Độ nổi chi tiết từ ảnh',
                               info='Cộng hoa văn ước lượng từ ảnh vào mặt trước. 0 là tắt. Trên 1,5 dễ sinh vết gấp.')
            texture = gr.Checkbox(value=True, label='Tạo texture UV', info='Màu 2K ở mức Chi tiết và Tối đa. GLB lưu kèm texture.')
            normal_detail = gr.Checkbox(value=True, label='Chi tiết bằng normal map + bóng hốc',
                                        info='Bake chi tiết lưới đầy đủ và vân khắc từ ảnh vào texture. Thêm ~7 giây.')
            photo_texture = gr.Checkbox(value=True, label='Chiếu màu ảnh gốc lên mặt trước',
                                        info='Màu mặt trước lấy thẳng từ ảnh nên sắc nét hơn. Mặt sau vẫn do AI tô.')
            with gr.Accordion('Điều chỉnh bề mặt', open=False):
                threshold = gr.Slider(5, 50, value=25, step=1, label='Ngưỡng bề mặt',
                                      info='Giảm nhẹ nếu chi tiết mảnh bị mất. Ngưỡng thấp có thể làm bề mặt dày hơn.')
                seed = gr.Number(value=1234, precision=0, label='Seed của Hunyuan',
                                 info='Đổi số khác để Hunyuan đoán lại phần khuất nếu chưa ưng.')
                smooth = gr.Checkbox(value=True, label='Làm mượt nốt sần',
                                     info='Áp dụng trước khi cộng chi tiết từ ảnh.')
                smoothing_method = gr.Radio(list(SMOOTHING_METHODS), value=SMOOTHING_METHODS[1],
                                            label='Bộ lọc làm mượt', info='Dùng khi bật Làm mượt nốt sần.')
                meshlab_angle = gr.Slider(0, 90, value=45, step=1, label='Góc giữ cạnh MeshLab (độ)',
                                          info='Cạnh có góc lớn hơn ngưỡng được giữ. Ngưỡng cao làm mượt nhiều vùng hơn.')
                meshlab_iterations = gr.Slider(1, 5, value=1, step=1, label='Số lượt lọc MeshLab')
                measure_deviation = gr.Checkbox(value=True, label='Đo độ lệch bề mặt bằng MeshLab',
                                                info='Ước lượng từ các đỉnh tới bề mặt theo hai chiều, trước và sau làm mượt cùng Spot.')
                mesh_cleanup = gr.Checkbox(value=True, label='Làm sạch lưới bằng Open3D',
                                           info='Dọn đỉnh và mặt trùng, mặt suy biến, tính lại pháp tuyến trước khi tạo UV.')
                halfmesh_mode = gr.Radio(list(HALFMESH_MODES), value=HALFMESH_MODES[1],
                                         label='halfmesh — sửa cấu trúc lưới',
                                         info='Sửa lỗi kết nối trước Spot. Chọn vá lỗ khi cần, vì khoảng trống thật cũng có thể bị lấp.')
                max_hole_edges = gr.Slider(3, 128, value=12, step=1, label='Số cạnh tối đa của lỗ cần vá',
                                           info='Chỉ dùng khi bật vá lỗ. Giá trị nhỏ giới hạn các lỗ được xử lý.')
                spot_mode = gr.Radio(list(SPOT_MODES), value=SPOT_MODES[1], label='Spot — điểm nghi lỗi',
                                     info='Chỉ phát hiện đánh dấu điểm nhọn để kiểm tra. Sửa có thể ảnh hưởng hoa văn thật.')
                spot_threshold = gr.Slider(1, 20, value=3.5, step=0.5, label='Ngưỡng phát hiện Spot',
                                          info='Tăng để đánh dấu ít điểm hơn. Ngưỡng cao vẫn có thể nhận nhầm chi tiết thật.')
                spot_max_displacement = gr.Slider(0.05, 1, value=0.25, step=0.05,
                                                 label='Giới hạn dịch chuyển Spot',
                                                 info='Chỉ dùng khi sửa, tính theo bán kính vùng lân cận. Mặc định 0,25.')
                processed = gr.Image(label='Ảnh sau tách nền và căn chỉnh', interactive=False)
            button = gr.Button('Tạo mô hình 3D', variant='primary')
        with gr.Column():
            view_mode = gr.Radio(list(VIEW_FILES), value='Đất sét', label='Cách xem',
                                 info='Bản xem Spot đánh dấu đỏ các đỉnh nghi lỗi trước bước cộng chi tiết và tạo UV.')
            viewer = gr.HTML(value=viewer_html())
            existing_glb = gr.File(label='Mở GLB có sẵn để xem', file_types=['.glb'], type='filepath')
            status = gr.Textbox(label='Kết quả', interactive=False)
            files = gr.Files(label='Lưu mô hình', interactive=False)
    gr.Examples(examples=[[str(ROOT / 'TripoSR/examples/chair.png')]], inputs=[image], label='Ảnh mẫu')
    result_folder = gr.State(None)
    button.click(convert, [image, remove_background, shape_model, quality, texture, threshold, relief,
                           smooth, photo_texture, seed, normal_detail, view_mode, mesh_cleanup,
                           spot_mode, spot_threshold, spot_max_displacement, halfmesh_mode, max_hole_edges,
                           smoothing_method, meshlab_angle, meshlab_iterations, measure_deviation],
                 [viewer, files, status, processed, result_folder], concurrency_limit=1, api_name='create_model')
    view_mode.change(change_view, [view_mode, result_folder], viewer, queue=False)
    existing_glb.upload(open_glb, existing_glb, [viewer, result_folder, view_mode], queue=False)


def run(inbrowser=True, port=7860):
    import uvicorn
    import webbrowser
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    app.mount('/viewer', StaticFiles(directory=str(ROOT / 'viewer')), name='viewer')
    app = gr.mount_gradio_app(app, demo.queue(max_size=4), path='/',
                              allowed_paths=[str(ROOT / 'results')])
    if inbrowser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    uvicorn.run(app, host='127.0.0.1', port=port)

if __name__ == '__main__':
    run()
