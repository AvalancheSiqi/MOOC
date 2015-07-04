//
//  MemeDetailViewController.swift
//  MemeMe
//
//  Created by SiqiWu on 2/07/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import UIKit

class MemeDetailViewController: UIViewController {

    @IBOutlet weak var memedImageView: UIImageView!
    var memedImage: UIImage!
    
    
    // - LifeCycle
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewWillAppear(animated: Bool) {
        super.viewWillAppear(animated)
        self.tabBarController?.tabBar.hidden = true
        memedImageView.image = memedImage
    }
    
    override func viewDidAppear(animated: Bool) {
        super.viewWillDisappear(animated)
        self.tabBarController?.tabBar.hidden = false
    }
}
