This is how you'd load an alignn model:
if reverse_label:
        from alignn.models.alignn import ALIGNN
        model = ALIGNN(config=config.model)
        model_path = (
    "alignn/pretrain_test/checkpoint_177.pt"
                    )
        model.load_state_dict(torch.load(model_path, map_location=device)["model"])