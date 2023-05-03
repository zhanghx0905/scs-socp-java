import os

for size in [5000]:
    for seed in [1]:
        os.system(f"python ./scripts/lasso_problem.py --seed {seed} --size {size}")
    os.system(f'grep "2000" -rl ./src/test/java/com/nmsolver/LassoTest.java | xargs sed -i "s/2000/{size}/g"')
    os.system("mvn -Dtest=LassoTest test")
    os.system(f'grep "{size}" -rl ./src/test/java/com/nmsolver/LassoTest.java | xargs sed -i "s/{size}/2000/g"')
