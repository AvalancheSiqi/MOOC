//
//  MemeCollectionViewController.swift
//  MemeMe
//
//  Created by SiqiWu on 2/07/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class MemeCollectionViewController: UIViewController, UICollectionViewDelegate, UICollectionViewDataSource {
    
    var memes: [Meme]!
    @IBOutlet var collectionView: UICollectionView!
    
    
    // - LifeCycle
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
        
        let applicationDelegate = UIApplication.sharedApplication().delegate as! AppDelegate
        memes = applicationDelegate.memes
        
        self.collectionView.reloadData()
    }
    

    // - CollectionView DataSource and Delegate
    func collectionView(collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return self.memes.count
    }
    
    func collectionView(collectionView: UICollectionView, cellForItemAtIndexPath indexPath: NSIndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCellWithReuseIdentifier("MemeCollectionViewCell", forIndexPath: indexPath) as! MemeCollectionViewCell
        let meme = self.memes[indexPath.item]
        
        cell.memedImage.image = meme.memedImage
        return cell
    }
    
    func collectionView(collectionView: UICollectionView, didSelectItemAtIndexPath indexPath: NSIndexPath) {
        let memeDetailVC = self.storyboard!.instantiateViewControllerWithIdentifier("MemeDetailViewController") as! MemeDetailViewController
        
        memeDetailVC.memedImage = self.memes[indexPath.item].memedImage
        memeDetailVC.hidesBottomBarWhenPushed = true
        self.navigationController!.pushViewController(memeDetailVC, animated: true)
    }
    
    
    // - Add new meme
    @IBAction func addMeme(sender: AnyObject) {
        let memeVC = self.storyboard!.instantiateViewControllerWithIdentifier("MemeViewController") as! MemeViewController
        self.presentViewController(memeVC, animated: true, completion: nil)
    }
    
}
