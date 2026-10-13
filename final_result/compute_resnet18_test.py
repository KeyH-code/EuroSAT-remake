"""对预先锁定的 ResNet18 checkpoint 进行一次最终 test 评价。"""

from compute_resnet18_validation import main


if __name__ == "__main__":
    main(split="test")
