import torch
import guided_transfer_learning as gtl
import unittest


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 15)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(15, 5)
        self.linear3 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        return x


class TestCreateScoutData(unittest.TestCase):
    def setUp(self):
        # seed everything
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        orig_model = TinyModel().to(device)
        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.SGD(orig_model.parameters(), lr=0.05)

        def train_model(model, x, y, epochs, guidance_matrix=None):
            for _ in range(epochs):
                result = model(x)
                loss = loss_function(
                    result, y
                )  # torch.tensor([1.00, 2.00], dtype=torch.float)
                model.zero_grad()
                loss.backward()
                if guidance_matrix is not None:
                    gtl.apply_guidance(model, guidance_matrix)
                optimizer.step()

        base_train_x = torch.tensor(
            [[0.1, 0.2], [0.11, 0.22], [0.3, 0.4], [0.3, 0.4]]
        ).to(device)
        base_train_y = torch.tensor(
            [[1.1, 2.2], [1.1, 2.2], [1.3, 2.4], [0.3, 0.4]]
        ).to(device)

        train_model(model=orig_model, x=base_train_x, y=base_train_y, epochs=10)

        model = TinyModel().to(device)
        model.load_state_dict(orig_model.state_dict())

        scout_indexes = [[0, 1], [2, 3]]

        scout_data_x = gtl.create_scout_data_from_ranged_indexes(
            scout_indexes, base_train_x
        )
        scout_data_y = gtl.create_scout_data_from_ranged_indexes(
            scout_indexes, base_train_y
        )

        my_scouts = gtl.create_scouts(model, should_save_guidance=False)
        for s in range(len(scout_data_x)):
            model.load_state_dict(orig_model.state_dict())
            train_model(model=model, x=scout_data_x[s], y=scout_data_y[s], epochs=10)
            my_scouts.add_scout(model)

        guiding_matrix = my_scouts.create_raw_guidance(device)
        del my_scouts

        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.ROOM,
            should_save_guidance=False,
        )
        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.CHILLY,
            should_save_guidance=False,
        )
        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.FREEZING,
            should_save_guidance=False,
        )
        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.WARM,
            should_save_guidance=False,
        )
        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.EVAPORATING,
            should_save_guidance=False,
        )
        gtl.adjust_guidance_matrix(
            guiding_matrix,
            gtl.Focus.ZERO_ENFORCED_AND_NORMALIZED,
            gtl.Temp.ICY,
            should_save_guidance=False,
        )

        train_model(
            model=model,
            x=base_train_x,
            y=base_train_y,
            epochs=10,
            guidance_matrix=guiding_matrix,
        )
        self.state_dict = model.state_dict()

    def test_end_model_weights_biases(self):
        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear1.weight"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor(
                    [
                        [-0.0053, 0.3793],
                        [-0.5817, -0.5199],
                        [-0.2723, 0.1896],
                        [-0.0140, 0.5607],
                        [-0.0628, 0.1871],
                        [-0.2137, -0.1390],
                        [-0.6755, -0.4683],
                        [-0.2912, 0.0266],
                        [0.2816, 0.4273],
                        [-0.4773, -0.3048],
                        [0.2553, 0.5849],
                        [-0.1455, 0.5291],
                        [-0.1113, 0.0789],
                        [0.6403, -0.6560],
                        [-0.4452, -0.1790],
                    ]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear1.bias"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor(
                    [
                        -0.2756,
                        0.6125,
                        -0.4583,
                        -0.3255,
                        -0.4940,
                        -0.6622,
                        -0.4128,
                        0.6091,
                        0.3255,
                        0.3531,
                        0.0297,
                        -0.3625,
                        0.1329,
                        -0.6602,
                        -0.5109,
                    ]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear2.weight"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor(
                    [
                        [
                            -0.1331,
                            0.1520,
                            0.1514,
                            -0.1145,
                            -0.0093,
                            0.1651,
                            0.2567,
                            0.0844,
                            0.0185,
                            0.1681,
                            -0.1607,
                            0.0481,
                            -0.2042,
                            -0.1790,
                            -0.1334,
                        ],
                        [
                            0.1168,
                            0.1248,
                            -0.1529,
                            0.0780,
                            0.1417,
                            -0.0326,
                            0.0099,
                            0.0948,
                            0.1919,
                            0.2577,
                            -0.1823,
                            -0.0946,
                            0.1093,
                            0.2139,
                            0.2247,
                        ],
                        [
                            0.2278,
                            0.0514,
                            -0.2245,
                            0.0238,
                            -0.1615,
                            -0.2406,
                            0.2294,
                            0.1963,
                            -0.2576,
                            0.0483,
                            -0.0435,
                            -0.0425,
                            -0.1182,
                            0.0993,
                            -0.1529,
                        ],
                        [
                            0.0947,
                            0.1301,
                            0.1848,
                            0.0965,
                            -0.2555,
                            -0.1675,
                            0.1289,
                            0.0533,
                            -0.2021,
                            -0.1489,
                            0.2425,
                            0.1740,
                            -0.1127,
                            -0.0650,
                            -0.2460,
                        ],
                        [
                            -0.0046,
                            -0.1944,
                            -0.1992,
                            -0.0142,
                            0.0388,
                            -0.1057,
                            0.1532,
                            -0.1571,
                            0.2343,
                            0.1769,
                            -0.2177,
                            -0.0643,
                            0.0117,
                            0.0377,
                            0.0612,
                        ],
                    ]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear2.bias"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor([0.0688, 0.0783, -0.1260, 0.1208, -0.2477]),
            )
        )

        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear3.weight"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor(
                    [
                        [-0.2651, -0.1120, -0.2178, -0.1565, -0.3665],
                        [-0.0111, 0.1935, -0.2913, 0.0360, 0.3201],
                    ]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                torch.round(self.state_dict["linear3.bias"].to(device="cpu") * 10**4)
                / 10**4,
                torch.Tensor([-0.0460, 0.5124]),
            )
        )
