//
//  RecordedAudio.swift
//  Pitch Perfect
//
//  Created by SiqiWu on 25/06/2015.
//  Copyright (c) 2015 Siqi. All rights reserved.
//

import Foundation

class RecordedAudio: NSObject {
    var filePathUrl: NSURL!
    var title: String!
    
    init(filePathUrl: NSURL, title: String) {
        self.filePathUrl = filePathUrl
        self.title = title
    }
}