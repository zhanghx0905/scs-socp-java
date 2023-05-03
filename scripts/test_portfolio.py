import os

for size in [500]:
    for seed in range(10):
        os.system(f"python ./scripts/portfolio_problem.py --seed {seed} --size {size}")
    os.system(f'grep "2000" -rl ./src/test/java/com/nmsolver/PortfolioTest.java | xargs sed -i "s/2000/{size}/g"')
    os.system("mvn -Dtest=PortfolioTest test")
    os.system(f'grep "{size}" -rl ./src/test/java/com/nmsolver/PortfolioTest.java | xargs sed -i "s/{size}/2000/g"')
