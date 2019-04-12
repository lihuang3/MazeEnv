# MazeEnv
Maze environment based on OpenAI gym

To render animation in RND/ICM algos, make the following changes in MazeEnv render()

```python
        #==================================================================================
        os.makedirs(ROOT_DIR+'/pics', exist_ok=True)
        filename = ROOT_DIR+'/pics/'+'fig'+'%05d'%(1+len(os.listdir(ROOT_DIR+'/pics')))+'.png'
        plt.savefig(filename)
        plt.clf()
        #==================================================================================

        # plt.show(False)
        # plt.pause(0.0001)

```

