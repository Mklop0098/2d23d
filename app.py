import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import logging
from pathlib import Path
from html import escape
from urllib.parse import quote, urlencode
import gradio as gr
from engine import ROOT, QUALITY, SHAPE_MODELS, generate


def viewer_html(path=None, clay=False):
    source = '/viewer/index.html'
    if path:
        source += '?' + urlencode({'model': '/file=' + quote(str(Path(path).resolve()).replace('\\', '/'), safe='/:'),
                                    'clay': int(clay)})
    return (f'<iframe src="{escape(source, quote=True)}" title="Xem mô hình bằng Three.js" '
            'style="width:100%;height:650px;border:0;border-radius:10px" allow="fullscreen"></iframe>')


def convert(image, remove_background, shape_model, quality, texture, threshold, relief, smooth,
            photo_texture, seed, normal_detail, view_mode, mesh_cleanup=True, progress=gr.Progress()):
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
                                   mesh_cleanup=mesh_cleanup)
    except Exception as error:
        logging.exception('Không thể tạo mô hình')
        raise gr.Error(f'Không thể tạo mô hình. {error}') from error
    status = (f"{metrics['shape_model']} · Hoàn thành trong {metrics['seconds']:.1f} giây · "
              f"RAM tiến trình cao nhất {metrics['peak_process_ram_gib']:.2f} GiB · "
              f"{metrics['faces']:,} mặt lưới · Lưới {metrics['resolution']} · "
              f"Texture {metrics['texture_resolution']} px")
    preview = folder / ('geometry.glb' if view_mode == 'Đất sét' else 'model.glb')
    return viewer_html(preview), [str(folder / 'model.glb'), str(folder / 'model.obj')], status, str(folder / 'input.png'), str(folder)


def change_view(view_mode, folder):
    if not folder:
        return gr.skip()
    if Path(folder).suffix.lower() == '.glb':
        return viewer_html(folder, clay=view_mode == 'Đất sét')
    return viewer_html(Path(folder) / ('geometry.glb' if view_mode == 'Đất sét' else 'model.glb'))


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
                                     info='Làm mượt nhẹ trước khi cộng chi tiết. Hunyuan dùng mức nhẹ hơn để giữ hoa văn.')
                mesh_cleanup = gr.Checkbox(value=True, label='Làm sạch lưới bằng Open3D',
                                           info='Dọn đỉnh và mặt trùng, mặt suy biến, tính lại pháp tuyến trước khi tạo UV.')
                processed = gr.Image(label='Ảnh sau tách nền và căn chỉnh', interactive=False)
            button = gr.Button('Tạo mô hình 3D', variant='primary')
        with gr.Column():
            view_mode = gr.Radio(['Đất sét', 'Có màu'], value='Đất sét', label='Cách xem',
                                 info='Đất sét giúp kiểm tra hình khối và hoa văn.')
            viewer = gr.HTML(value=viewer_html())
            existing_glb = gr.File(label='Mở GLB có sẵn để xem', file_types=['.glb'], type='filepath')
            status = gr.Textbox(label='Kết quả', interactive=False)
            files = gr.Files(label='Lưu mô hình', interactive=False)
    gr.Examples(examples=[[str(ROOT / 'TripoSR/examples/chair.png')]], inputs=[image], label='Ảnh mẫu')
    result_folder = gr.State(None)
    button.click(convert, [image, remove_background, shape_model, quality, texture, threshold, relief,
                           smooth, photo_texture, seed, normal_detail, view_mode, mesh_cleanup],
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
