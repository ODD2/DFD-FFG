from src.dataset.ffpp import FFPPDataModule


if __name__ == "__main__":
    class Dummy():
        pass
    model = Dummy()
    model.n_px = 224
    model.transform = lambda x: x
    dtm = FFPPDataModule(
        ["REAL", "DF"],
        ["c23"],
        clip_duration=3,
        num_frames=10,
        vid_ext=".avi",
        data_dir="datasets/ffpp/",
        num_workers=0,
        batch_size=1
    )
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtl = dtm.train_dataloader()
    i = 0
    for b in dtl:
        i += 1
        print(b[-2])
        if (i == 10):
            break

    print("asdfasdfasdfasdf")
    i = 0
    for b in dtl:
        i += 1
        print(b[-2])
        if (i == 10):
            break
