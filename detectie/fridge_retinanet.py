from icevision.all import *

url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
dest_dir = "fridge"
data_dir = icedata.load_data(url, dest_dir)

class_map = ClassMap(["milk_bottle", "carton", "can", "water_bottle"])
parser = parsers.voc(annotations_dir=data_dir / "odFridgeObjects/annotations",
                     images_dir=data_dir / "odFridgeObjects/images",
                     class_map=class_map)
# Records
train_records, valid_records = parser.parse()

train_dl = retinanet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = retinanet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

model = retinanet.model(num_classes=len(class_map))

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

class LightModel(retinanet.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-2)

light_model = LightModel(model, metrics=metrics)

trainer = pl.Trainer(max_epochs=40, gpus=1)
trainer.fit(light_model, train_dl, valid_dl)
