"""Обновить описание после реального запуска, не подставляя числа заранее."""
import json
from pathlib import Path

base=Path(__file__).resolve().parent
from app.app import build_app
from PIL import Image
interface=build_app()
example=next((base/'data/raw').glob('*/*.jpg'))
with Image.open(example) as image:
    prediction,timing=interface.fn(image)
assert len(prediction)==3 and abs(sum(prediction.values())-1)<1e-5
(base/'models/app_check.json').write_text(json.dumps({'example':example.name,'prediction':prediction,'timing':timing,'interface_created':True},ensure_ascii=False,indent=2),encoding='utf-8')
interface.close()
meta=json.loads((base/'models/metadata.json').read_text(encoding='utf-8'))
rows=[]
for p in sorted((base/'runs').glob('*/summary.json')):
    cfg=json.loads((p.parent/'config.json').read_text(encoding='utf-8'))
    summary=json.loads(p.read_text(encoding='utf-8'))
    test=p.parent/'test_metrics.json'
    metrics=json.loads(test.read_text(encoding='utf-8')) if test.exists() else None
    rows.append((p.parent.name,cfg,summary,metrics))
lines=['## Результаты выполненных экспериментов','',
       '| Запуск | lr | Validation accuracy | Test accuracy | Время, с | Параметры |',
       '|---|---:|---:|---:|---:|---:|']
for name,cfg,summary,metrics in rows:
    test=f"{metrics['accuracy']:.3f}" if metrics else 'не оценивалась'
    lines.append(f"| {name} | {cfg['lr']} | {summary['best_val_accuracy']:.3f} | {test} | {summary['seconds']:.1f} | {summary['parameters']:,} |")
winner=next(r for r in rows if r[0]==meta['selected_run'])
lines.extend(['', f"По validation выбрана модель **{winner[1]['model']}**, запуск `{winner[0]}`. "
              f"Её test accuracy составила **{winner[3]['accuracy']:.3f}** на 27 снимках.", '',
              'Набор небольшой: одна ошибка на test меняет accuracy примерно на 3.7 процентного пункта. '
              'Поэтому различие в одну фотографию не даёт надёжного доказательства превосходства архитектуры.', ''])
for name,cfg,summary,metrics in rows:
    if metrics:
        cm=metrics['confusion_matrix']
        errors=[(cm[i][j],meta['classes'][i],meta['classes'][j]) for i in range(len(cm)) for j in range(len(cm)) if i!=j and cm[i][j]]
        if errors:
            count,actual,pred=max(errors)
            lines.append(f'Для `{name}` самая частая путаница: {actual} → {pred}, {count} снимков. Примеры ошибок показаны в ноутбуке.')
        else:
            lines.append(f'Для `{name}` на этой тестовой выборке ошибок нет. Это не гарантирует правильную классификацию других снимков.')
        hist=json.loads((base/'runs'/name/'history.json').read_text(encoding='utf-8'))
        if hist[-1]['train_loss']<hist[0]['train_loss'] and hist[-1]['val_loss']>min(h['val_loss'] for h in hist[:-1]):
            lines.append('К последней эпохе train loss уменьшилась, а validation loss выше достигнутого минимума. '
                         'Это повод остановиться на сохранённом checkpoint; по короткой кривой нельзя уверенно установить причину роста.')
lines.extend(['', 'Время измерено на одной машине и включает обучение и validation, но не скачивание весов. '
              'Число параметров описывает размер модели, а не напрямую её задержку в приложении.', '',
              'Ограничения: открытые фотографии вместо самостоятельной съёмки, один seed, небольшой test, '
              'возможные похожие изображения, отсутствие проверки на другом устройстве. Для улучшения оценки '
              'нужны собственные новые снимки и несколько повторов разбиения.'])
report='\n'.join(lines)+'\n'
(base/'reports/results.md').write_text(report,encoding='utf-8')
readme=base.parent/'README.md'
text=readme.read_text(encoding='utf-8')
start=text.index('**Текущий статус:**')
end=text.index('\n\n## Структура',start)
text=text[:start]+'**Текущий статус:** четыре эксперимента выполнены, тестовые метрики рассчитаны,\nлучшая модель экспортирована в ONNX и проверена на test. Данные открытые;\nтребование самостоятельной съёмки не выполнено.'+text[end:]
text=text.replace('Зависимости зафиксированы по версиям, но установка в текущей среде не проверена.',
                  'Зависимости зафиксированы по версиям; эксперименты выполнены в среде, описанной в логах запусков.')
text=text.replace('1. Добавьте минимум 30 своих изображений в каждую из трёх папок классов в\n   `experiments/data/raw/`. Заполните описание реального сбора в `data/README.md`.',
                  '1. В `experiments/data/raw/` уже включены 180 открытых фотографий.\n   Происхождение, отбор и ограничения описаны в `experiments/data/README.md`.')
text=text.replace('Для сдачи нужно добавить реальные результаты, выполненный notebook, графики\nи описание собственного датасета, затем создать публичный репозиторий GitHub.',
                  'Для сдачи необходимо согласовать использование открытого набора вместо собственной съёмки.\nРезультаты и выполненный notebook включены в репозиторий.')
text+='\n'+report
readme.write_text(text,encoding='utf-8')
nb_path=base/'notebooks/experiments.ipynb'
nb=json.loads(nb_path.read_text(encoding='utf-8'))
nb['cells'][0]['source']=['# Fine-Tuning: сравнение двух моделей\n\n',
    'Сравниваю две предобученные архитектуры на подвыборке Flowers: 180 открытых фотографий, три класса. '
    'Снимки взяты из внешнего набора, а не сделаны мной. Это ограничение относительно условия лабораторной.\n']
nb['cells'][2]['source']=['## Данные\n\nТри класса: daisy, roses, sunflowers. По 60 снимков, общий сплит 126/27/27. '
    'Исходные файлы сохранены вместе с лицензией, авторами и хешами. Сбор — скачивание и воспроизводимый отбор из TensorFlow Flowers.\n']
nb['cells'][-1]['source']=(report+'\n## Приложение\n\nЗапуск из корня: `python -m app.app`.\n').splitlines(True)
nb_path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print(report)
